"""Normalizing-flow prior backed by a trained flowjax model.

`jim-nf` (`jimgw.cli_nf`) fits a flowjax masked-autoregressive spline flow to GWTC
posterior samples and serializes it. This module reconstructs that flow and exposes
it as a Jim [`Prior`][jimgw.core.prior.Prior] so an existing GW event's posterior can
act as an informed, differentiable joint prior for downstream parameter estimation.

Importing this module pulls in flowjax/JAX, so CLI code should import it lazily (inside
builder functions) to keep `--help` fast.
"""

import json
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import Affine, Invert, RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed
from flowjax.flows import masked_autoregressive_flow
from jaxtyping import Array, Float, Key

from jimgw.core.prior import Prior
from jimgw.typing import FloatScalar

# Maps common pesummary/GWTC posterior field names to jim parameter names. Used by
# `jim-nf` at training time so the serialized flow already carries jim names.
PESUMMARY_TO_JIM = {
    "chirp_mass": "M_c",
    "mass_ratio": "q",
    "luminosity_distance": "d_L",
    "iota": "iota",
    "ra": "ra",
    "dec": "dec",
    "psi": "psi",
    "phase": "phase_c",
    "geocent_time": "t_c",
    "spin_1z": "s1_z",
    "spin_2z": "s2_z",
}

# Current flowjax `masked_autoregressive_flow` defaults, pinned explicitly so a flow
# saved today reloads correctly even if flowjax changes its defaults later.
FLOW_LAYERS = 8
NN_WIDTH = 50
NN_DEPTH = 1

# Rejection sampling against the optional box bounds: how much to oversample each round
# and how many rounds before giving up (see `NormalizingFlowPrior._sample_in_bounds`).
DRAW_OVERSAMPLE = 2
MAX_REJECTION_ROUNDS = 20


def build_inner_flow(key: Key, n_dim: int, knots: int, interval: float):
    """The bare masked-autoregressive spline flow, before the standardization wrap."""
    return masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(n_dim)),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        flow_layers=FLOW_LAYERS,
        nn_width=NN_WIDTH,
        nn_depth=NN_DEPTH,
    )


def build_flow_skeleton(key: Key, n_dim: int, knots: int, interval: float):
    """A flow with the same structure as a serialized model, for deserialization.

    The serialized flow is the inner flow wrapped by ``Invert(Affine(...))`` (the
    standardization preprocess). The Affine loc/scale here are placeholders; their
    leaves are overwritten by ``eqx.tree_deserialise_leaves``.
    """
    inner = build_inner_flow(key, n_dim, knots, interval)
    return Transformed(inner, Invert(Affine(jnp.zeros(n_dim), jnp.ones(n_dim))))


def load_flow(model_path: str | Path, meta: dict):
    """Deserialize a trained flow from ``model_path`` using its metadata dict."""
    skeleton = build_flow_skeleton(
        jax.random.key(meta["seed"]),
        meta["n_dim"],
        meta["knots"],
        meta["interval"],
    )
    return eqx.tree_deserialise_leaves(str(model_path), skeleton)


class NormalizingFlowPrior(Prior):
    """Joint prior whose density is a trained flowjax normalizing flow.

    Parameters are sampled and evaluated directly in physical space (the flow was
    trained on physical posterior samples). Optional per-parameter box bounds return
    ``-inf`` outside the physical range, guarding samplers against unphysical proposals.
    """

    flow: Transformed
    lower: Float[Array, " n_dims"]
    upper: Float[Array, " n_dims"]
    bounded: bool = eqx.field(static=True)

    @property
    def is_normalized(self) -> bool:
        # The flow is a proper normalized density. The optional box bounds truncate a
        # small amount of tail mass without renormalizing, so strictly the density
        # integrates to slightly less than 1; the deficit is far below the precision
        # that matters for the evidence computed by SMC/NSS. `sample` rejects that
        # truncated mass so draws and `log_prob` support stay consistent.
        return True

    def __repr__(self):
        return f"NormalizingFlowPrior(parameter_names={self.parameter_names})"

    def __init__(
        self,
        flow: Transformed,
        parameter_names: list[str],
        bounds: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """
        Args:
            flow: The loaded flowjax distribution (physical space).
            parameter_names: jim parameter names, in the flow's dimension order.
            bounds: Optional ``{name: (min, max)}`` physical box; parameters absent
                from the dict are unbounded.
        """
        super().__init__(parameter_names)
        self.flow = flow
        bounds = bounds or {}
        self.lower = jnp.array(
            [bounds.get(name, (-jnp.inf, jnp.inf))[0] for name in parameter_names]
        )
        self.upper = jnp.array(
            [bounds.get(name, (-jnp.inf, jnp.inf))[1] for name in parameter_names]
        )
        self.bounded = bool(
            jnp.any(jnp.isfinite(self.lower) | jnp.isfinite(self.upper))
        )

    def log_prob(self, z: dict[str, Float]) -> FloatScalar:
        x = jnp.stack([z[name] for name in self.parameter_names])
        base = self.flow.log_prob(x)
        in_bounds = jnp.all((x >= self.lower) & (x <= self.upper))
        return jnp.where(in_bounds, base, -jnp.inf)

    def _sample_in_bounds(
        self, rng_key: Key, n_samples: int
    ) -> Float[Array, "n_samples n_dims"]:
        """Draw ``n_samples`` flow samples inside the box, by rejection.

        The flow has support on all of R^n, so a few draws per thousand land outside a
        hard physical edge the training posterior piles up against (q > 1, d_L < 0).
        Returning those would hand the sampler particles whose log-prior is -inf, which
        SMC never culls (its weights come from the likelihood alone), leaving them frozen
        at their initial position and resampled straight into the output.
        """
        collected = []
        n_found = 0
        for _ in range(MAX_REJECTION_ROUNDS):
            rng_key, draw_key = jax.random.split(rng_key)
            x = self.flow.sample(draw_key, (DRAW_OVERSAMPLE * (n_samples - n_found),))
            keep = x[jnp.all((x >= self.lower) & (x <= self.upper), axis=1)]
            collected.append(keep)
            n_found += keep.shape[0]
            if n_found >= n_samples:
                return jnp.concatenate(collected)[:n_samples]
        raise RuntimeError(
            f"Rejection sampling the NF prior gave only {n_found}/{n_samples} in-bounds "
            f"draws after {MAX_REJECTION_ROUNDS} rounds. The bounds likely exclude most "
            "of the flow's mass; check that they match the range the flow was trained on."
        )

    def sample(
        self, rng_key: Key, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        if self.bounded:
            samples = self._sample_in_bounds(rng_key, n_samples)
        else:
            samples = self.flow.sample(rng_key, (n_samples,))
        return {name: samples[:, i] for i, name in enumerate(self.parameter_names)}


def load_nf_prior(
    model_path: str | Path,
    metadata_path: Optional[str | Path] = None,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> NormalizingFlowPrior:
    """Load a serialized flow and its metadata into a `NormalizingFlowPrior`.

    ``metadata_path`` defaults to ``model_path`` with a ``.json`` suffix. The stored
    ``parameters`` are already jim names (converted at training time).
    """
    model_path = Path(model_path)
    if metadata_path is None:
        metadata_path = model_path.with_suffix(".json")
    meta = json.loads(Path(metadata_path).read_text())
    flow = load_flow(model_path, meta)
    return NormalizingFlowPrior(flow, meta["parameters"], bounds)
