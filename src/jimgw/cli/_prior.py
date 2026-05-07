import logging

import numpy as np

from jimgw.cli._config import (
    CosineSpec,
    GaussianSpec,
    PowerLawSpec,
    PriorConfig,
    RayleighSpec,
    SamplingConfig,
    SineSpec,
    UniformSpec,
    UniformSphereSpec,
)
from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    GaussianPrior,
    PowerLawPrior,
    RayleighPrior,
    SinePrior,
    UniformPrior,
    UniformSpherePrior,
)
from jimgw.core.constants import C_SI
from jimgw.core.single_event.detector import GroundBased2G

logger = logging.getLogger(__name__)


def build_prior(cfg: PriorConfig):
    """Build a CombinePrior from the dict-keyed prior config."""
    components = []
    for name, spec in cfg.root.items():
        if isinstance(spec, UniformSpec):
            components.append(UniformPrior(spec.min, spec.max, [name]))
        elif isinstance(spec, GaussianSpec):
            components.append(GaussianPrior(spec.loc, spec.scale, [name]))
        elif isinstance(spec, SineSpec):
            components.append(SinePrior([name]))
        elif isinstance(spec, CosineSpec):
            components.append(CosinePrior([name]))
        elif isinstance(spec, PowerLawSpec):
            components.append(PowerLawPrior(spec.min, spec.max, spec.alpha, [name]))
        elif isinstance(spec, RayleighSpec):
            components.append(RayleighPrior(spec.scale, [name]))
        elif isinstance(spec, UniformSphereSpec):
            components.append(UniformSpherePrior([name]))
        else:
            raise ValueError(f"Unknown prior spec type for '{name}': {type(spec)}")

    prior = CombinePrior(components)
    logger.info(
        "Built prior: %d parameter(s): %s",
        len(prior.parameter_names),
        prior.parameter_names,
    )
    return prior


def _max_ifo_delay(ifos: list[GroundBased2G]) -> float:
    """Maximum light travel time from geocenter to any IFO in the network."""
    return max(float(np.linalg.norm(ifo.vertex)) / C_SI for ifo in ifos)


def adapt_prior_for_ns_time(
    prior_cfg: PriorConfig,
    trigger_time: float,
    ifos: list[GroundBased2G],
    sampling_cfg: SamplingConfig,
) -> PriorConfig | None:
    """For NS-AW: adjust time parameters so the unit-cube bounds are exact.

    NS-AW requires every sampling-space parameter to lie in [0, 1].
    Two cases require adaptation:

    1. **t_c in prior, time_frame != "geocentric"**: the ``t_c → t_det`` sample
       transform shifts by a sky-dependent delay, so the unit-cube bounds for
       ``t_det`` cannot be exact.  Fix: replace ``t_c`` with ``t_det`` using
       widened absolute GPS bounds:

           t_det ∈ [trigger + t_c.min - max_delay, trigger + t_c.max + max_delay]

    2. **t_det in prior, time_frame = "geocentric"**: the user wants to sample in
       ``t_c`` but the prior is on ``t_det`` (absolute GPS).  The reverse shift
       is also sky-dependent.  Fix: replace ``t_det`` with ``t_c`` using widened
       bounds:

           t_c ∈ [t_det.min - max_delay, t_det.max + max_delay]

    In both cases ``max_delay`` is the maximum light travel time to any IFO in
    the network, and the reverse time transform is added as a likelihood
    transform so the likelihood still receives the correct parameter.

    Returns the modified :class:`PriorConfig`, or ``None`` if no adaptation is
    needed.
    """
    has_t_c = "t_c" in prior_cfg.root
    has_t_det = "t_det" in prior_cfg.root

    # Case 2: t_det in prior + geocentric sampling → adapt to t_c prior.
    if has_t_det and sampling_cfg.time_frame == "geocentric":
        t_det_spec = prior_cfg.root["t_det"]
        if not isinstance(t_det_spec, UniformSpec):
            raise ValueError(
                "NS-AW sampler: the 't_det' prior must be 'uniform' for automatic "
                "conversion to 't_c'. Either use a uniform t_det prior or replace "
                "'t_det' with 't_c' in [prior] with relative bounds."
            )
        max_delay = _max_ifo_delay(ifos)
        lo = t_det_spec.min - max_delay
        hi = t_det_spec.max + max_delay
        logger.warning(
            "NS-AW sampler: replacing t_det ~ Uniform(%.6f, %.6f) in [prior] with "
            "t_c ~ Uniform(%.6f, %.6f) (max IFO light-travel delay = %.2f ms).",
            t_det_spec.min,
            t_det_spec.max,
            lo,
            hi,
            max_delay * 1e3,
        )
        new_root = {
            ("t_c" if k == "t_det" else k): (
                UniformSpec(min=lo, max=hi) if k == "t_det" else v
            )
            for k, v in prior_cfg.root.items()
        }
        return PriorConfig.model_validate(new_root)

    # No t_c in prior, or user already opted for geocentric t_c sampling → no change.
    if not has_t_c or sampling_cfg.time_frame == "geocentric":
        return None

    # Case 1: t_c in prior + detector time_frame → adapt to t_det prior.
    t_c_spec = prior_cfg.root["t_c"]
    if not isinstance(t_c_spec, UniformSpec):
        raise ValueError(
            "NS-AW sampler: the 't_c' prior must be 'uniform' for automatic "
            "conversion to 't_det'. Either use a uniform t_c prior, set "
            "[sampling] time_frame = 'geocentric' to sample t_c directly, or "
            "replace 't_c' with 't_det' in [prior] and provide absolute GPS bounds."
        )

    max_delay = _max_ifo_delay(ifos)
    lo = trigger_time + t_c_spec.min - max_delay
    hi = trigger_time + t_c_spec.max + max_delay

    logger.warning(
        "NS-AW sampler: replacing t_c ~ Uniform(%.4f, %.4f) in [prior] with "
        "t_det ~ Uniform(%.6f, %.6f) (max IFO light-travel delay = %.2f ms). "
        "To sample t_c directly instead, set [sampling] time_frame = 'geocentric'.",
        t_c_spec.min,
        t_c_spec.max,
        lo,
        hi,
        max_delay * 1e3,
    )

    # Rebuild the prior dict, preserving insertion order, substituting t_c → t_det.
    new_root = {
        ("t_det" if k == "t_c" else k): (
            UniformSpec(min=lo, max=hi) if k == "t_c" else v
        )
        for k, v in prior_cfg.root.items()
    }
    return PriorConfig.model_validate(new_root)
