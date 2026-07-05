import logging

from jimgw.cli._prior import build_prior
from jimgw.cli_dp._config import LikelihoodConfig
from jimgw.core.single_event.detector import QuantumSensor
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.marginalization_config import (
    DistanceMargConfig,
    PhaseMargConfig,
    TimeMargConfig,
)

logger = logging.getLogger(__name__)


def build_likelihood(
    cfg: LikelihoodConfig,
    sensors: list[QuantumSensor],
    waveform,
    trigger_time: float,
) -> TransientLikelihoodFD:
    """Build a plain TransientLikelihoodFD from the validated likelihood config.

    Unlike `jimgw.cli._likelihood.build_likelihood`, there is no
    heterodyne/multiband branch — see `LikelihoodConfig` for why.
    """
    phase_marg = PhaseMargConfig() if cfg.phase_marginalization else None

    time_marg = None
    if cfg.time_marginalization is not None:
        time_marg = TimeMargConfig(tc_range=cfg.time_marginalization.tc_range)

    dist_marg = None
    if cfg.distance_marginalization is not None:
        dist_combined = build_prior(cfg.distance_marginalization.distance_prior)
        dist_marg = DistanceMargConfig(
            distance_prior=dist_combined.base_prior[0],
            n_dist_points=cfg.distance_marginalization.n_dist_points,
            ref_dist=cfg.distance_marginalization.ref_dist,
        )

    likelihood = TransientLikelihoodFD(
        detectors=sensors,
        waveform=waveform,
        fixed_parameters=cfg.fixed_parameters if cfg.fixed_parameters else None,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        trigger_time=trigger_time,
        phase_marginalization=phase_marg,
        time_marginalization=time_marg,
        distance_marginalization=dist_marg,
    )
    logger.info(
        "Built likelihood: f_min=%.1f, f_max=%.1f, trigger_time=%.3f",
        cfg.f_min,
        cfg.f_max,
        trigger_time,
    )
    return likelihood
