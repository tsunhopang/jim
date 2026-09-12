"""Unit tests for cli_dp real-noise signal injection helpers."""

import jax
import jax.numpy as jnp
import pytest

from jimgw.cli_dp._config import InjectionConfig
from jimgw.cli_dp._data import (
    _inject_signal,
    _resolve_injection_parameters,
    _trial_optimal_snr,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_quantum_sensor_preset
from jimgw.core.single_event.waveform import RippleDarkPhotonWaveform, RippleIMRPhenomD

F_MIN, F_MAX = 20.0, 512.0
DURATION = 8.0
SAMPLING_FREQUENCY = 1024.0
TRIGGER_TIME = 1343651418.0
SEG_START = TRIGGER_TIME - DURATION + 2.0
LAMBDA_REF = 2.14e4

BASE_PARAMETERS = {
    "M_c": 30.0,
    "eta": 0.25,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "sigma_1": 0.2,
    "sigma_2": -0.2,
    "Lambda": 2.14e4,
}

WAVEFORM = RippleDarkPhotonWaveform(RippleIMRPhenomD(f_ref=F_MIN))


def make_sensor(name="QS-I", seed=0):
    qs = get_quantum_sensor_preset()[name]
    qs.tau_Xe = 56.98406646
    qs.freq_Xe = 10.05450347

    n_times = int(round(DURATION * SAMPLING_FREQUENCY))
    noise = jax.random.normal(jax.random.PRNGKey(seed), (n_times,)) * 1e-20
    qs.set_data(
        Data(
            td=noise,
            delta_t=1.0 / SAMPLING_FREQUENCY,
            start_time=SEG_START,
            name=f"{qs.name}_real",
        )
    )
    frequencies = jnp.fft.rfftfreq(n_times, d=1.0 / SAMPLING_FREQUENCY)
    psd_values = jnp.full(frequencies.shape, 1e-40)
    qs.set_psd(
        PowerSpectrum(name=f"{qs.name}_psd", values=psd_values, frequencies=frequencies)
    )
    return qs


class TestInjectSignal:
    def test_adds_signal_on_top_of_real_data(self):
        qs = make_sensor()
        real_fd_before = qs.data.fft()

        params = {**BASE_PARAMETERS, "d_L": 400.0}
        optimal_snr = _inject_signal(qs, params, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME)

        assert optimal_snr > 0
        assert jnp.isfinite(optimal_snr)
        assert not jnp.allclose(qs.data.fft(), real_fd_before)

    def test_logs_warning(self, caplog):
        qs = make_sensor()
        params = {**BASE_PARAMETERS, "d_L": 400.0}
        with caplog.at_level("WARNING", logger="jimgw.cli_dp._data"):
            _inject_signal(qs, params, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME)
        assert any(
            "SIMULATED SIGNAL INJECTED" in record.message for record in caplog.records
        )


class TestResolveInjectionParameters:
    def test_direct_distance_used_as_is(self):
        qs = make_sensor()
        injection = InjectionConfig(
            injection_parameters={**BASE_PARAMETERS, "d_L": 123.0}
        )
        resolved = _resolve_injection_parameters(
            [qs], injection, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME
        )
        assert resolved["d_L"] == 123.0

    def test_target_snr_calibrates_network_distance(self):
        qs1 = make_sensor("QS-I", seed=1)
        qs2 = make_sensor("QS-II", seed=2)
        target_snr = 25.0
        injection = InjectionConfig(
            injection_parameters=dict(BASE_PARAMETERS),
            target_optimal_snr=target_snr,
        )
        resolved = _resolve_injection_parameters(
            [qs1, qs2], injection, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME
        )
        network_snr_sq = sum(
            _trial_optimal_snr(qs, WAVEFORM, resolved, F_MIN, F_MAX, TRIGGER_TIME) ** 2
            for qs in [qs1, qs2]
        )
        assert network_snr_sq**0.5 == pytest.approx(target_snr, rel=1e-6)

    def test_lambda_ratio_is_rescaled_to_lambda(self):
        qs = make_sensor()
        params = {k: v for k, v in BASE_PARAMETERS.items() if k != "Lambda"}
        injection = InjectionConfig(
            injection_parameters={**params, "d_L": 123.0, "Lambda_ratio": 2.0},
        )
        resolved = _resolve_injection_parameters(
            [qs], injection, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME, LAMBDA_REF
        )
        assert "Lambda_ratio" not in resolved
        assert resolved["Lambda"] == pytest.approx(2.0 * LAMBDA_REF)

    def test_lambda_ratio_is_rescaled_before_snr_calibration(self):
        qs = make_sensor()
        params = {k: v for k, v in BASE_PARAMETERS.items() if k != "Lambda"}
        injection = InjectionConfig(
            injection_parameters={**params, "Lambda_ratio": 1.0},
            target_optimal_snr=25.0,
        )
        resolved = _resolve_injection_parameters(
            [qs], injection, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME, LAMBDA_REF
        )
        assert resolved["Lambda"] == pytest.approx(LAMBDA_REF)
        assert _trial_optimal_snr(
            qs, WAVEFORM, resolved, F_MIN, F_MAX, TRIGGER_TIME
        ) == pytest.approx(25.0, rel=1e-6)

    def test_lambda_ratio_without_reference_is_rejected(self):
        qs = make_sensor()
        params = {k: v for k, v in BASE_PARAMETERS.items() if k != "Lambda"}
        injection = InjectionConfig(
            injection_parameters={**params, "d_L": 123.0, "Lambda_ratio": 1.0},
        )
        with pytest.raises(ValueError, match="no reference cutoff"):
            _resolve_injection_parameters(
                [qs], injection, WAVEFORM, F_MIN, F_MAX, TRIGGER_TIME
            )
