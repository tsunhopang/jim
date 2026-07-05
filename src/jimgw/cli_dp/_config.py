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


class DataConfig(BaseModel):
    """Load a real analysis segment and PSD from quantum-sensor .mat files.

    No signal injection: this runs the likelihood directly on the real
    segment (a search/PE run), unlike the reference example script which
    injects a synthetic signal for pipeline validation.
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
