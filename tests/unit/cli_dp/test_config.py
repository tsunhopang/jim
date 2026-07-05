"""Unit tests for jim-dp-run config schema, focused on injection opt-in."""

import pytest
from pydantic import ValidationError

from jimgw.cli_dp._config import DataConfig, InjectionConfig

_DATA_BASE = dict(
    sensors=["QS-I"],
    sensor_files={"QS-I": "real_data/Sensor1.mat"},
    data_start_gps=1343650218.0,
    trigger_time=1343651418.0,
    duration=8.0,
)


def test_injection_defaults_to_none():
    cfg = DataConfig(**_DATA_BASE)
    assert cfg.injection is None


def test_injection_with_distance():
    injection = InjectionConfig(injection_parameters={"d_L": 400.0})
    assert injection.target_optimal_snr is None


def test_injection_with_target_snr():
    injection = InjectionConfig(injection_parameters={}, target_optimal_snr=30.0)
    assert injection.target_optimal_snr == 30.0


def test_injection_rejects_both_distance_and_target_snr():
    with pytest.raises(ValidationError):
        InjectionConfig(injection_parameters={"d_L": 400.0}, target_optimal_snr=30.0)


def test_injection_rejects_neither_distance_nor_target_snr():
    with pytest.raises(ValidationError):
        InjectionConfig(injection_parameters={})


def test_injection_forbids_unknown_keys():
    with pytest.raises(ValidationError):
        InjectionConfig(injection_parameters={"d_L": 400.0}, extra_field=1)


def test_data_config_with_injection_round_trips():
    raw = {
        **_DATA_BASE,
        "injection": {
            "injection_parameters": {"d_L": 400.0},
        },
    }
    cfg = DataConfig(**raw)
    assert cfg.injection is not None
    assert cfg.injection.injection_parameters["d_L"] == 400.0
