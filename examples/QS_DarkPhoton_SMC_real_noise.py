"""Dark-photon parameter estimation on real Xe-comagnetometer sensor noise.

Loads real time-series data from the five Xe-comagnetometer sensors
(Sensor1..5.mat), estimates each sensor's PSD via Welch's method on the real
data, injects a dark-photon signal directly into an 8 s real-noise segment
around the trigger time, and samples M_c, sigma_1, sigma_2, ra, dec, and
eps_BD with the BlackJAX SMC sampler.
"""

import time
from pathlib import Path

import corner
import numpy as np
import scipy.io
from scipy.signal import welch
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, CosinePrior, UniformPrior
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import (
    get_QS_I,
    get_QS_II,
    get_QS_III,
    get_QS_IV,
    get_QS_V,
)
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import complex_inner_product, inner_product
from jimgw.core.single_event.waveform import RippleDarkPhotonWaveform, RippleIMRPhenomD
from jimgw.samplers.config import BlackJAXSMCConfig

TARGET_NETWORK_SNR = 30.0
OUTDIR = Path(__file__).parent
REAL_DATA_DIR = Path(__file__).resolve().parents[3] / "real_data"

f_min = 20.0
f_max = 512.0
duration = 8.0
sampling_frequency = 1024.0
post_trigger_duration = 2.0

# Data start 2022-08-04 20:10:00 CST, trigger 2022-08-04 20:30:00 CST
# (Hefei/Hangzhou local time), converted to GPS via astropy.time.Time.
DATA_START_GPS = 1343650218.0
TRIGGER_GPS = 1343651418.0

seg_start = TRIGGER_GPS - duration + post_trigger_duration
n_times = int(round(duration * sampling_frequency))
injection_frequencies = jnp.fft.rfftfreq(n_times, d=1.0 / sampling_frequency)

# --- Sensor setup: Sensor{i}.mat <-> get_QS_{roman(i)}() ---

get_qs_by_sensor = [get_QS_I, get_QS_II, get_QS_III, get_QS_IV, get_QS_V]
quantum_sensors = []
real_segments = {}

for i, get_qs in enumerate(get_qs_by_sensor, start=1):
    qs = get_qs()
    mat = scipy.io.loadmat(REAL_DATA_DIR / f"Sensor{i}.mat")
    sig = mat["Sig"].flatten()
    t = mat["t"].flatten()
    fs = 1.0 / (t[1] - t[0])
    qs.freq_Xe = float(mat["freq"].item())
    qs.tau_Xe = float(mat["T2"].flat[0])

    # PSD estimated from the full real time series (project convention).
    f, psd_vals = welch(sig, fs=fs, window=("tukey", 0.25), nperseg=int(8 * fs))  # type: ignore[arg-type]
    measured_psd = PowerSpectrum(
        name=f"{qs.name}_welch_psd",
        values=jnp.asarray(psd_vals),
        frequencies=jnp.asarray(f),
    ).interpolate(injection_frequencies)
    qs.set_psd(measured_psd)

    # Real 8 s analysis segment around the trigger time, demeaned to avoid
    # the sensor's large DC offset leaking into the FFT.
    idx0 = int(round((seg_start - DATA_START_GPS) * fs))
    td_segment = sig[idx0 : idx0 + n_times]
    td_segment = td_segment - td_segment.mean()

    quantum_sensors.append(qs)
    real_segments[qs.name] = td_segment

    print(f"{qs.name}: freq_Xe={qs.freq_Xe:.4f} Hz, tau_Xe={qs.tau_Xe:.4f} s")

# --- Waveform model ---

waveform = RippleDarkPhotonWaveform(RippleIMRPhenomD(f_ref=f_min))

# --- Injection parameters (likelihood space) ---
# eps_BD and d_L both scale amplitude multiplicatively, so the choice of
# injected eps_BD is arbitrary; d_L is calibrated below to hit the target SNR.

injection_parameters = {
    "M_c": 30.0,
    "eta": 0.25,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "d_L": 400.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "sigma_1": 0.2,
    "sigma_2": -0.2,
    "eps_BD": 0.5,
}

print("The trial injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")

# --- Calibrate d_L to hit a target network SNR against the real PSDs ---
# (SNR scales as 1/d_L, so a single trial injection is enough to rescale.)

network_snr_sq = 0.0
for qs in quantum_sensors:
    qs.inject_signal(
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=TRIGGER_GPS,
        waveform_model=waveform,
        parameters=injection_parameters,
        f_min=f_min,
        f_max=f_max,
        zero_noise=True,
    )
    network_snr_sq += float(qs.optimal_snr) ** 2

network_snr_trial = network_snr_sq**0.5
injection_parameters["d_L"] *= network_snr_trial / TARGET_NETWORK_SNR
print(f"Calibrated d_L = {injection_parameters['d_L']:.2f} Mpc")

print("The final injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")


# --- Inject the calibrated signal into the real noise segments ---


def inject_into_real_noise(qs, td_segment, parameters):
    """Inject a dark-photon signal on top of a real time-domain noise segment."""
    params = parameters.copy()
    params["trigger_time"] = float(TRIGGER_GPS)
    params["gmst"] = float(compute_gmst(TRIGGER_GPS))

    real_data = Data(
        td=jnp.asarray(td_segment),
        delta_t=1.0 / sampling_frequency,
        start_time=seg_start,
        name=f"{qs.name}_real",
    )
    qs.set_data(real_data)
    qs.set_frequency_bounds(f_min, f_max)

    band_frequencies = qs.sliced_frequencies
    polarisations = waveform(band_frequencies, params)
    band_signal = qs.fd_response(band_frequencies, polarisations, params)

    real_fd = qs.data.fft()
    strain_data = real_fd.at[jnp.nonzero(qs.frequency_mask)[0]].add(band_signal)

    qs.set_data(
        Data.from_fd(
            fd_strain=strain_data,
            frequencies=qs.frequencies,
            start_time=seg_start,
            name=f"{qs.name}_injected",
        )
    )
    qs.set_frequency_bounds()

    df = qs.sliced_frequencies[1] - qs.sliced_frequencies[0]
    optimal_snr = inner_product(band_signal, band_signal, qs.sliced_psd, df) ** 0.5
    match_filtered_snr = (
        complex_inner_product(band_signal, qs.sliced_fd_data, qs.sliced_psd, df)
        / optimal_snr
    )
    qs.optimal_snr = optimal_snr
    qs.match_filtered_snr = match_filtered_snr


network_snr_sq = 0.0
for qs in quantum_sensors:
    inject_into_real_noise(qs, real_segments[qs.name], injection_parameters)
    network_snr_sq += float(qs.optimal_snr) ** 2
    print(
        f"{qs.name}: optimal SNR = {float(qs.optimal_snr):.2f}, "
        f"matched-filter SNR = {complex(qs.match_filtered_snr):.2f}"
    )
print(f"Network optimal SNR: {network_snr_sq**0.5:.2f}")

# --- Prior: M_c, sigma_1, sigma_2, ra, dec, eps_BD ---

prior = CombinePrior(
    [
        UniformPrior(25.0, 35.0, parameter_names=["M_c"]),
        UniformPrior(-0.5, 0.5, parameter_names=["sigma_1"]),
        UniformPrior(-0.5, 0.5, parameter_names=["sigma_2"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, 1.0, parameter_names=["eps_BD"]),
    ]
)

# --- Everything else fixed to its injected value ---

fixed_parameters = {
    key: value
    for key, value in injection_parameters.items()
    if key not in prior.parameter_names
}

# --- Likelihood ---

likelihood = TransientLikelihoodFD(
    quantum_sensors,
    waveform=waveform,
    fixed_parameters=fixed_parameters,
    trigger_time=TRIGGER_GPS,
    f_min=f_min,
    f_max=f_max,
)

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=5000,
        n_mcmc_steps_per_dim=100,
        target_ess=10000,
        initial_cov_scale=0.5,
        target_acceptance_rate=0.234,
        scale_adaptation_gain=3.0,
        persistent_sampling=True,
        temperature_ladder=None,
    ),
    periodic={"ra": (0.0, 2 * jnp.pi)},
    verbose=True,
)

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Results ---

diagnostics = jim.get_diagnostics()
print(f"log Z = {diagnostics['log_Z']:.2f}")
print(f"Likelihood evaluations: {diagnostics['n_likelihood_evaluations']:,}")

chains = jim.get_samples()

samples_path = OUTDIR / "QS_DarkPhoton_SMC_real_noise_samples.npz"
np.savez(samples_path, **{k: np.asarray(v) for k, v in chains.items()})
print(f"Saved samples to {samples_path}")

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "sigma_1": r"$\sigma_1$",
    "sigma_2": r"$\sigma_2$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "eps_BD": r"$\epsilon_{\rm BD}$",
}

truths = [float(injection_parameters[k]) for k in jim.prior.parameter_names]

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    truths=truths,
    truth_color="#DA5B2A",
    color="#3B4CC0",
    bins=40,
    smooth=0.9,
    quantiles=[0.16, 0.5, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    max_n_ticks=4,
    show_titles=True,
    title_fmt=".2f",
    use_math_text=True,
    label_kwargs={"fontsize": 16},
    title_kwargs={"fontsize": 14},
)
fig.savefig(OUTDIR / "QS_DarkPhoton_SMC_real_noise.pdf", bbox_inches="tight")
