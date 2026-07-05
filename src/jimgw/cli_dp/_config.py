"""Pydantic config models for the jim-dp-run CLI pipeline.

JAX-free like `jimgw.cli._config`, so `jim-dp-run --help` starts in
milliseconds. Waveform, prior, and output models carry no GW-specific
assumptions, so they are reused directly from `jimgw.cli._config`.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from jimgw.cli._config import (
    CLIDistanceMargConfig,
    CLITimeMargConfig,
    OutputConfig,
    PriorConfig,
    WaveformConfig,
)
from jimgw.samplers.config import SamplerConfig

_SUPPORTED_SENSORS = frozenset({"QS-I", "QS-II", "QS-III", "QS-IV", "QS-V"})


class InjectionConfig(BaseModel):
    """Simulated dark-photon signal added on top of the real noise segment.

    Off by default (`data.injection` unset): PE then runs directly on the
    real segment, as before. When set, a deterministic signal computed from
    `injection_parameters` (likelihood-space parameter names, e.g. M_c, eta,
    s1_z, s2_z, t_c, phase_c, iota, ra, dec, psi, sigma_1, sigma_2, eps_BD,
    plus either `d_L` or `target_optimal_snr` -- see below) is added to each
    sensor's real frequency-domain strain.

    Exactly one of `injection_parameters["d_L"]` or `target_optimal_snr` must
    be given: either inject at a fixed luminosity distance, or let d_L be
    solved for so the injected signal's network optimal SNR (summed over all
    configured sensors) hits `target_optimal_snr`.
    """

    model_config = {"extra": "forbid"}
    injection_parameters: dict[str, float]
    target_optimal_snr: Optional[float] = None

    @model_validator(mode="after")
    def _check_distance_xor_target_snr(self) -> "InjectionConfig":
        has_d_l = "d_L" in self.injection_parameters
        has_target = self.target_optimal_snr is not None
        if has_d_l == has_target:
            raise ValueError(
                "Provide exactly one of injection_parameters['d_L'] or "
                "target_optimal_snr, not both or neither."
            )
        return self


class DataConfig(BaseModel):
    """Load a real analysis segment and PSD from quantum-sensor .mat files.

    By default no signal is injected and the likelihood runs directly on the
    real segment (a search/PE run). Set `injection` to additionally add a
    simulated dark-photon signal on top of the real segment for pipeline
    validation.
    """

    model_config = {"extra": "forbid"}

    type: Literal["mat_segment"] = "mat_segment"
    sensors: list[str]
    sensor_files: dict[str, Path]  # sensor name -> .mat file path
    data_start_gps: float
    """GPS time of the start of each sensor's raw recording (assumed
    synchronized across all listed sensors)."""
    trigger_time: float
    duration: float
    post_trigger_duration: float = 2.0
    psd_nperseg_duration: float = 8.0
    """Welch nperseg, in seconds, for PSD estimation from the full raw series."""
    sig_key: str = "Sig"
    t_key: str = "t"
    T2_key: str = "T2"
    freq_key: str = "freq"
    injection: Optional[InjectionConfig] = None

    @field_validator("sensors")
    @classmethod
    def _check_sensors(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("data.sensors must be a non-empty list")
        unknown = [s for s in v if s not in _SUPPORTED_SENSORS]
        if unknown:
            raise ValueError(
                f"Unknown sensor name(s): {unknown}. "
                f"Supported: {sorted(_SUPPORTED_SENSORS)}"
            )
        if len(v) != len(set(v)):
            duplicates = [s for s in set(v) if v.count(s) > 1]
            raise ValueError(f"Duplicate sensor name(s): {duplicates}")
        return v

    @model_validator(mode="after")
    def _check_all_sensors_have_files(self) -> "DataConfig":
        missing = [s for s in self.sensors if s not in self.sensor_files]
        if missing:
            raise ValueError(f"sensor_files missing for: {missing}")
        return self


class LikelihoodConfig(BaseModel):
    """Likelihood config, trimmed from `jimgw.cli._config.LikelihoodConfig`.

    Omits `heterodyne`/`multiband`: their frequency-banding formulas assume
    compact-binary-only phasing and have not been validated for the
    dark-photon dephasing term.
    """

    model_config = {"extra": "forbid"}
    f_min: float
    f_max: float
    fixed_parameters: dict[str, float] = Field(default_factory=dict)
    phase_marginalization: bool = False
    time_marginalization: Optional[CLITimeMargConfig] = None
    distance_marginalization: Optional[CLIDistanceMargConfig] = None


class PipelineConfig(BaseModel):
    model_config = {"extra": "forbid"}

    seed: int = 0
    data: DataConfig
    waveform: WaveformConfig
    prior: PriorConfig
    likelihood: LikelihoodConfig
    sampler: SamplerConfig
    output: OutputConfig
