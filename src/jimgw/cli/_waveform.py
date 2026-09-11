import logging

from jimgw.cli._config import WaveformConfig
from jimgw.core.single_event.waveform import (
    RippleDarkPhotonWaveform,
    RippleIMRPhenomD,
    RippleIMRPhenomD_NRTidalv2,
    RippleIMRPhenomHM,
    RippleIMRPhenomPv2,
    RippleIMRPhenomXAS,
    RippleIMRPhenomXAS_NRTidalv3,
    RippleIMRPhenomXHM,
    RippleIMRPhenomXP,
    RippleIMRPhenomXPHM,
    RippleScalarWaveform,
    RippleSineGaussian,
    RippleTaylorF2,
)

logger = logging.getLogger(__name__)

_REGISTRY = {
    "TaylorF2": RippleTaylorF2,
    "IMRPhenomD": RippleIMRPhenomD,
    "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
    "IMRPhenomHM": RippleIMRPhenomHM,
    "IMRPhenomPv2": RippleIMRPhenomPv2,
    "IMRPhenomXAS": RippleIMRPhenomXAS,
    "IMRPhenomXAS_NRTidalv3": RippleIMRPhenomXAS_NRTidalv3,
    "IMRPhenomXHM": RippleIMRPhenomXHM,
    "IMRPhenomXP": RippleIMRPhenomXP,
    "IMRPhenomXPHM": RippleIMRPhenomXPHM,
    "SineGaussian": RippleSineGaussian,
    "DarkPhotonWaveform": RippleDarkPhotonWaveform,
    "ScalarWaveform": RippleScalarWaveform,
}


def build_waveform(cfg: WaveformConfig):
    """Instantiate the ripple waveform specified by *cfg*."""
    if cfg.approximant == "DarkPhotonWaveform":
        assert cfg.base_approximant is not None
        base_cls = _REGISTRY[cfg.base_approximant]
        base_waveform = base_cls(f_ref=cfg.f_ref)
        waveform = RippleDarkPhotonWaveform(base_waveform)
        logger.info(
            "Built waveform: DarkPhotonWaveform(base=%s(f_ref=%.1f))",
            type(base_waveform).__name__,
            cfg.f_ref,
        )
        return waveform

    if cfg.approximant == "ScalarWaveform":
        assert cfg.base_approximant is not None
        assert cfg.scalar_power is not None
        base_cls = _REGISTRY[cfg.base_approximant]
        base_waveform = base_cls(f_ref=cfg.f_ref)
        waveform = RippleScalarWaveform(base_waveform, k=cfg.scalar_power)
        logger.info(
            "Built waveform: ScalarWaveform(base=%s(f_ref=%.1f), k=%d)",
            type(base_waveform).__name__,
            cfg.f_ref,
            cfg.scalar_power,
        )
        return waveform

    cls = _REGISTRY[cfg.approximant]
    waveform = cls(f_ref=cfg.f_ref)
    logger.info("Built waveform: %s(f_ref=%.1f)", type(waveform).__name__, cfg.f_ref)
    return waveform
