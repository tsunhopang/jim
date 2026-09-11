import jax

import numpy as np
import jax.numpy as jnp
from itertools import combinations
from pathlib import Path
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import get_ET, get_H1, get_quantum_sensor_preset
from jimgw.core.constants import (
    C_SI,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
)
from jimgw.core.single_event.polarization import rotated_wave_basis
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from tests.utils import assert_all_finite, assert_all_in_range

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

GPS_TIME = 1126259462.0
DURATION = 4.0
F_MIN, F_MAX = 20.0, 1024.0
SAMPLING_FREQUENCY = F_MAX * 2

# Likelihood-space (fully expanded) parameters used as the reference injection.
REFERENCE_PARAMS = {
    "M_c": 28.0,
    "eta": 0.24,
    "s1_x": 0.3,
    "s1_y": 0.2,
    "s1_z": 0.1,
    "s2_x": -0.1,
    "s2_y": 0.2,
    "s2_z": -0.3,
    "d_L": 440.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "t_c": 0.0,
}


def make_detector():
    det = get_H1()
    psd = PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
    det.set_psd(psd)
    return det


def inject_reference(det, trigger_time=GPS_TIME, **overrides):
    """Inject the reference signal (zero noise) into *det*."""
    params = {**REFERENCE_PARAMS, **overrides}
    det.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=trigger_time,
        waveform_model=RippleIMRPhenomD(f_ref=20.0),
        parameters=params,
        f_min=F_MIN,
        f_max=F_MAX,
        zero_noise=True,
    )


# ---------------------------------------------------------------------------
# inject_signal tests
# ---------------------------------------------------------------------------


class TestInjectSignal:
    """Tests for inject_signal: core behavior and the transform pipeline."""

    # ------------------------------------------------------------------
    # Core behavior
    # ------------------------------------------------------------------

    def test_zero_noise_creates_data(self):
        """Data object is populated after a zero-noise injection."""
        det = make_detector()
        inject_reference(det)

        assert det.data is not None
        assert len(det.data.td) == int(DURATION * SAMPLING_FREQUENCY)
        assert det.data.start_time == GPS_TIME - DURATION + 2.0

    def test_zero_noise_signal_nonzero_in_band(self):
        """Injected signal is non-zero inside the frequency band."""
        det = make_detector()
        inject_reference(det)

        assert jnp.any(jnp.abs(det.sliced_fd_data) > 0)

    def test_zero_noise_frequency_bounds_respected(self):
        """Sliced frequencies lie within the requested band."""
        det = make_detector()
        inject_reference(det)

        assert_all_in_range(det.sliced_frequencies, F_MIN, F_MAX)

    def test_noisy_injection_differs_from_zero_noise(self):
        """Adding noise produces data that differs from the zero-noise case."""
        det_clean = make_detector()
        inject_reference(det_clean)

        det_noisy = make_detector()
        params = dict(REFERENCE_PARAMS)
        det_noisy.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=params,
            f_min=F_MIN,
            f_max=F_MAX,
            zero_noise=False,
            rng_key=jax.random.key(42),
        )

        assert not jnp.allclose(
            det_clean.sliced_fd_data,
            det_noisy.sliced_fd_data,
            rtol=1e-05,
            atol=1e-23,
        )


# ---------------------------------------------------------------------------
# ET geometry tests
# ---------------------------------------------------------------------------


class TestET:
    """Tests for get_ET(): geometric consistency of the triangular ET configuration."""

    ET_ARM_LENGTH_M = 1e4  # 10 km

    def setup_method(self):
        self.ifos = get_ET()

    def test_returns_three_detectors(self):
        """get_ET returns exactly three GroundBased2G instances."""
        assert len(self.ifos) == 3

    def test_detector_names(self):
        """Sub-detectors are named ET1, ET2, ET3 in order."""
        assert [ifo.name for ifo in self.ifos] == ["ET1", "ET2", "ET3"]

    def test_arm_opening_angle_is_60_degrees(self):
        """Each sub-detector has 60° (π/3) between its x and y arms."""
        for ifo in self.ifos:
            delta = ifo.yarm_azimuth - ifo.xarm_azimuth
            assert abs(delta - np.pi / 3) < 1e-10, (
                f"{ifo.name}: arm opening angle is {np.degrees(delta):.4f}°, expected 60°"
            )

    def test_arms_rotated_240_degrees_between_detectors(self):
        """Consecutive sub-detectors have arm azimuths rotated by 240° (4π/3 rad)."""
        rotation = (4 / 3) * np.pi
        for i in range(2):
            dx = self.ifos[i + 1].xarm_azimuth - self.ifos[i].xarm_azimuth
            dy = self.ifos[i + 1].yarm_azimuth - self.ifos[i].yarm_azimuth
            assert abs(dx - rotation) < 1e-10, (
                f"ET{i + 1}→ET{i + 2} xarm rotation: {dx:.6f} rad, expected {rotation:.6f} rad"
            )
            assert abs(dy - rotation) < 1e-10, (
                f"ET{i + 1}→ET{i + 2} yarm rotation: {dy:.6f} rad, expected {rotation:.6f} rad"
            )

    def test_vertex_separations_match_arm_length(self):
        """
        Haversine distance between every pair of ET vertex positions should
        equal the arm length (10 km) to within 50 m.

        This checks both the propagation formula and that the triangle closes,
        following the approach used in bilby's TriangularInterferometerTest.
        """
        # Use the same WGS-84 radius get_ET uses: computed at ET1's latitude
        # (the initial latitude, before any vertex propagation).
        _a = EARTH_SEMI_MAJOR_AXIS / 1e3
        _b = EARTH_SEMI_MINOR_AXIS / 1e3
        lat0 = float(self.ifos[0].latitude)
        R = (
            _a * _b / np.sqrt(_a**2 * np.sin(lat0) ** 2 + _b**2 * np.cos(lat0) ** 2)
        ) * 1e3
        for ifo_a, ifo_b in combinations(self.ifos, 2):
            lat1 = float(ifo_a.latitude)
            lon1 = float(ifo_a.longitude)
            lat2 = float(ifo_b.latitude)
            lon2 = float(ifo_b.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            assert abs(dist - self.ET_ARM_LENGTH_M) < 50.0, (
                f"{ifo_a.name}↔{ifo_b.name}: {dist:.0f} m "
                f"(expected ~{self.ET_ARM_LENGTH_M:.0f} m ± 50 m)"
            )


# ---------------------------------------------------------------------------
# QuantumSensor
# ---------------------------------------------------------------------------


class TestQuantumSensor:
    """Tests for QuantumSensor.fd_response's psi-rotated field projection."""

    def setup_method(self):
        self.qs = get_quantum_sensor_preset()["QS-I"]
        self.qs.tau_Xe = 56.98406646
        self.qs.freq_Xe = 10.05450347

    def _params(self, **overrides):
        params = {
            "ra": 1.5,
            "dec": 0.5,
            "psi": 0.3,
            "gmst": float(compute_gmst(GPS_TIME)),
            "trigger_time": GPS_TIME,
            "t_c": 0.0,
            "eps_BD": 1.0,
        }
        params.update(overrides)
        return params

    def _B_sky(self, n=32):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        B_p = jax.random.normal(k1, (n,)) + 1j * jax.random.normal(k1, (n,))
        B_c = jax.random.normal(k2, (n,)) + 1j * jax.random.normal(k2, (n,))
        return {"p": B_p, "c": B_c}

    def test_finite_output(self):
        frequency = jnp.linspace(5.0, 15.0, 32)
        B_sky = self._B_sky(32)
        out = self.qs.fd_response(frequency, B_sky, self._params())
        assert out.shape == frequency.shape
        assert_all_finite(out)

    def test_psi_changes_response(self):
        """Regression guard: psi must actually affect the projected signal."""
        frequency = jnp.linspace(5.0, 15.0, 32)
        B_sky = self._B_sky(32)
        out_psi0 = self.qs.fd_response(frequency, B_sky, self._params(psi=0.0))
        out_psi1 = self.qs.fd_response(frequency, B_sky, self._params(psi=jnp.pi / 4))
        assert not jnp.allclose(out_psi0, out_psi1)

    def test_matches_manual_projection(self):
        """fd_response's arm projection should match a hand-built B_vec via
        rotated_wave_basis."""
        frequency = jnp.linspace(5.0, 15.0, 8)
        B_sky = self._B_sky(8)
        params = self._params()

        m, n = rotated_wave_basis(
            params["ra"], params["dec"], params["psi"], params["gmst"]
        )
        B_vec = jnp.einsum("i,f->if", m, B_sky["p"]) + jnp.einsum(
            "i,f->if", n, B_sky["c"]
        )
        arm_x, arm_y = self.qs.arms
        B_x = jnp.einsum("i,if->f", arm_x, B_vec)
        B_y = jnp.einsum("i,if->f", arm_y, B_vec)
        tau_inv = 1.0 / self.qs.tau_Xe
        lorentzian = tau_inv / (2j * jnp.pi * (frequency - self.qs.freq_Xe) + tau_inv)
        time_shift = self.qs.delay_from_geocenter(
            params["ra"], params["dec"], params["gmst"]
        )
        time_shift += params["trigger_time"] - self.qs.start_time + params["t_c"]
        expected = (
            lorentzian
            * (B_y - 1j * B_x)
            * jnp.exp(-2j * jnp.pi * frequency * time_shift)
        )

        actual = self.qs.fd_response(frequency, B_sky, params)
        assert jnp.allclose(actual, expected)

    def _B_sky_scalar(self, n=32):
        key = jax.random.PRNGKey(1)
        B_s = jax.random.normal(key, (n,)) + 1j * jax.random.normal(key, (n,))
        return {"s": B_s}

    def test_scalar_finite_output(self):
        frequency = jnp.linspace(5.0, 15.0, 32)
        out = self.qs.fd_response(frequency, self._B_sky_scalar(32), self._params())
        assert out.shape == frequency.shape
        assert_all_finite(out)

    def test_scalar_is_psi_independent(self):
        """A longitudinal field has no polarization angle."""
        frequency = jnp.linspace(5.0, 15.0, 32)
        B_sky = self._B_sky_scalar(32)
        out_psi0 = self.qs.fd_response(frequency, B_sky, self._params(psi=0.0))
        out_psi1 = self.qs.fd_response(frequency, B_sky, self._params(psi=jnp.pi / 4))
        assert jnp.allclose(out_psi0, out_psi1)

    def test_scalar_matches_manual_projection(self):
        """The scalar branch projects along the line of sight, not the m/n basis."""
        frequency = jnp.linspace(5.0, 15.0, 8)
        B_sky = self._B_sky_scalar(8)
        params = self._params()

        sky = self.qs.sky_vector(params["ra"], params["dec"], params["gmst"])
        B_vec = jnp.einsum("i,f->if", sky, B_sky["s"])
        arm_x, arm_y = self.qs.arms
        B_x = jnp.einsum("i,if->f", arm_x, B_vec)
        B_y = jnp.einsum("i,if->f", arm_y, B_vec)
        tau_inv = 1.0 / self.qs.tau_Xe
        lorentzian = tau_inv / (2j * jnp.pi * (frequency - self.qs.freq_Xe) + tau_inv)
        scale_Rb = 2.42e-7 / 5.76e-4 / 100.0
        time_shift = self.qs.delay_from_geocenter(
            params["ra"], params["dec"], params["gmst"]
        )
        time_shift += params["trigger_time"] - self.qs.start_time + params["t_c"]
        expected = (lorentzian * (B_y - 1j * B_x) + 1j * scale_Rb * B_y) * jnp.exp(
            -2j * jnp.pi * frequency * time_shift
        )

        actual = self.qs.fd_response(frequency, B_sky, params)
        assert jnp.allclose(actual, expected)

    def test_sky_vector_points_at_source(self):
        """sky_vector is the unit vector delay_from_geocenter projects onto."""
        params = self._params()
        sky = self.qs.sky_vector(params["ra"], params["dec"], params["gmst"])
        assert jnp.allclose(jnp.linalg.norm(sky), 1.0)
        expected_delay = -jnp.dot(sky, self.qs.vertex) / C_SI
        actual_delay = self.qs.delay_from_geocenter(
            params["ra"], params["dec"], params["gmst"]
        )
        assert jnp.allclose(actual_delay, expected_delay)
