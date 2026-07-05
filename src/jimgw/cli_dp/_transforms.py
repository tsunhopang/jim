"""Transform/periodic-bound inference for the dark-photon CLI.

Dark-photon `ra`/`dec` are consumed directly by `QuantumSensor.fd_response` —
there is no detector-frame reparametrization the way GW sky position uses
`SkyFrameToDetectorFrameSkyPositionTransform`. So, unlike
`jimgw.cli._transforms`, no sample transforms are needed at all, and the only
likelihood transform is an optional `q -> eta` conversion.
"""

import logging

from jimgw.cli._config import PriorConfig, UniformSpec
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.core.transforms import NtoMTransform

logger = logging.getLogger(__name__)

# Angles that may be sampled directly and are periodic on their full range.
_PERIODIC_PARAMS = ("ra", "phase_c")


def infer_likelihood_transforms(prior_params: frozenset[str]) -> list[NtoMTransform]:
    """Infer likelihood transforms (prior space -> likelihood space)."""
    if "q" in prior_params:
        logger.debug("Added MassRatioToSymmetricMassRatioTransform")
        return [MassRatioToSymmetricMassRatioTransform]
    return []


def infer_periodic(prior_cfg: PriorConfig) -> dict[str, tuple[float, float]]:
    """Periodic bounds to pass to `Jim(periodic=...)` for angles sampled directly."""
    periodic: dict[str, tuple[float, float]] = {}
    for name in _PERIODIC_PARAMS:
        spec = prior_cfg.root.get(name)
        if isinstance(spec, UniformSpec):
            periodic[name] = (spec.min, spec.max)
    return periodic
