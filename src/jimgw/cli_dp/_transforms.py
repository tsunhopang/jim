"""Transform/periodic-bound inference for the dark-photon CLI.

Dark-photon `ra`/`dec` are consumed directly by `QuantumSensor.fd_response` —
there is no detector-frame reparametrization the way GW sky position uses
`SkyFrameToDetectorFrameSkyPositionTransform`. So, unlike
`jimgw.cli._transforms`, no sample transforms are needed at all; the likelihood
transforms are an optional `q -> eta` conversion and the `Lambda_ratio -> Lambda`
rescaling shared with the GW CLI.
"""

import logging

import jax.numpy as jnp

from jimgw.cli._config import PriorConfig, UniformSpec, WaveformConfig
from jimgw.cli._transforms import lambda_ratio_transform
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.core.transforms import NtoMTransform

logger = logging.getLogger(__name__)

# Angles that may be sampled directly, with the full range they are periodic on.
_NATURAL_PERIOD = {"ra": (0.0, 2 * jnp.pi), "phase_c": (0.0, 2 * jnp.pi)}
_PERIODIC_PARAMS = tuple(_NATURAL_PERIOD)


def infer_likelihood_transforms(
    prior_params: frozenset[str], waveform_cfg: WaveformConfig
) -> list[NtoMTransform]:
    """Infer likelihood transforms (prior space -> likelihood space)."""
    transforms: list[NtoMTransform] = []
    if "q" in prior_params:
        logger.debug("Added MassRatioToSymmetricMassRatioTransform")
        transforms.append(MassRatioToSymmetricMassRatioTransform)
    transforms.extend(lambda_ratio_transform(prior_params, waveform_cfg))
    return transforms


def infer_periodic(
    prior_cfg: PriorConfig, nf_prior=None
) -> dict[str, tuple[float, float]]:
    """Periodic bounds to pass to `Jim(periodic=...)` for angles sampled directly.

    Angles come either from the `[prior]` section or from a trained NF prior, which
    supplies them instead of `[prior]` (the two may not overlap). Note that the flow's
    density is not itself periodic-continuous: it lives on all of R^n, so its value just
    past the 0/2pi seam is an extrapolation rather than the density wrapping round from
    the other side. Wrapping is still better than leaving the seam uncrossable.
    """
    periodic: dict[str, tuple[float, float]] = {}
    for name in _PERIODIC_PARAMS:
        spec = prior_cfg.root.get(name)
        if isinstance(spec, UniformSpec):
            periodic[name] = (spec.min, spec.max)

    if nf_prior is not None:
        periodic.update(_infer_nf_periodic(nf_prior))
    return periodic


def _infer_nf_periodic(nf_prior) -> dict[str, tuple[float, float]]:
    """Periodic bounds for angles supplied by the NF prior.

    A parameter is only wrapped if its `[nf_prior.bounds]` box is absent or covers the
    full period; wrapping into a narrower box would land samples outside it, where the
    prior is -inf.
    """
    periodic: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(nf_prior.parameter_names):
        period = _NATURAL_PERIOD.get(name)
        if period is None:
            continue
        lo, hi = float(nf_prior.lower[i]), float(nf_prior.upper[i])
        unbounded = lo == -jnp.inf and hi == jnp.inf
        if unbounded or (
            jnp.isclose(lo, period[0]).item() and jnp.isclose(hi, period[1]).item()
        ):
            periodic[name] = period
        else:
            logger.warning(
                "NF parameter %r has bounds (%g, %g), narrower than its period "
                "(%g, %g); leaving it non-periodic because wrapping would push "
                "samples outside the bounds, where the prior is -inf.",
                name,
                lo,
                hi,
                *period,
            )
    return periodic
