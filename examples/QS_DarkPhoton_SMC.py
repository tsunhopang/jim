"""Dark-photon parameter estimation on a simulated white-noise quantum sensor.

Injects a dark-photon signal into a single QuantumSensor with a flat (white)
PSD, rescaled so the optimal SNR is ~30, then samples only M_c, sigma_1, and
sigma_2 with the BlackJAX SMC sampler. All other parameters are held fixed
at their injected values.
"""

import time
from pathlib import Path

import corner
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.detector import get_QS_I
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleDarkPhotonWaveform, RippleIMRPhenomD
from jimgw.samplers.config import BlackJAXSMCConfig

TARGET_OPTIMAL_SNR = 30.0

# --- Sensor setup ---

qs = get_QS_I()
qs.tau_Xe = 56.98406646
qs.freq_Xe = 10.05450347

f_min = 20.0
f_max = 512.0
duration = 4.0
sampling_frequency = 1024.0

# --- Waveform model ---

waveform = RippleDarkPhotonWaveform(RippleIMRPhenomD(f_ref=f_min))

# --- Injection parameters (likelihood space) ---

gps = time.time() - 1000

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
    "sigma_1": 0.1,
    "sigma_2": -0.1,
}

print("The injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")

# --- Inject with a placeholder flat PSD, then rescale to hit the target SNR ---

n_times = int(round(duration * sampling_frequency))
white_frequencies = jnp.fft.rfftfreq(n_times, d=1.0 / sampling_frequency)

placeholder_psd_value = 1e-40
qs.set_psd(
    PowerSpectrum(
        name="white_psd",
        values=placeholder_psd_value * jnp.ones_like(white_frequencies),
        frequencies=white_frequencies,
    )
)

qs.inject_signal(
    duration=duration,
    sampling_frequency=sampling_frequency,
    trigger_time=gps,
    waveform_model=waveform,
    parameters=injection_parameters,
    f_min=f_min,
    f_max=f_max,
    zero_noise=True,
)

# optimal_snr scales as 1/sqrt(psd_value) for a flat PSD, so rescale accordingly
scale = (float(qs.optimal_snr) / TARGET_OPTIMAL_SNR) ** 2
qs.set_psd(
    PowerSpectrum(
        name="white_psd",
        values=placeholder_psd_value * scale * jnp.ones_like(white_frequencies),
        frequencies=white_frequencies,
    )
)

qs.inject_signal(
    duration=duration,
    sampling_frequency=sampling_frequency,
    trigger_time=gps,
    waveform_model=waveform,
    parameters=injection_parameters,
    f_min=f_min,
    f_max=f_max,
    zero_noise=False,
    rng_key=jax.random.key(0),
)
print(f"Optimal SNR after PSD rescaling: {qs.optimal_snr:.2f}")

# --- Prior: only M_c, sigma_1, sigma_2 are sampled ---

prior = CombinePrior(
    [
        UniformPrior(25.0, 35.0, parameter_names=["M_c"]),
        UniformPrior(-0.5, 0.5, parameter_names=["sigma_1"]),
        UniformPrior(-0.5, 0.5, parameter_names=["sigma_2"]),
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
    [qs],
    waveform=waveform,
    fixed_parameters=fixed_parameters,
    trigger_time=gps,
    f_min=f_min,
    f_max=f_max,
)

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=10000,
        n_mcmc_steps_per_dim=10,
        target_ess_fraction=0.8,
    ),
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

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "sigma_1": r"$\sigma_1$",
    "sigma_2": r"$\sigma_2$",
}

truths = [float(injection_parameters[k]) for k in jim.prior.parameter_names]

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    truths=truths,
)
fig.savefig(Path(__file__).parent / "QS_DarkPhoton_SMC.png")
