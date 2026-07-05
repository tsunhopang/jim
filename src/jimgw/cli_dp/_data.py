import logging

import jax.numpy as jnp

from jimgw.cli_dp._config import DataConfig, InjectionConfig
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import QuantumSensor, get_quantum_sensor_preset
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import complex_inner_product, inner_product

logger = logging.getLogger(__name__)


def build_sensors(
    cfg: DataConfig,
    waveform,
    f_min: float,
    f_max: float,
) -> list[QuantumSensor]:
    """Load real analysis segments and PSDs for the configured quantum sensors.

    For each sensor: load the full raw .mat time series (also sets
    `tau_Xe`/`freq_Xe`), slice out and demean the real analysis segment around
    the trigger time, then estimate the PSD via Welch on the full series with
    that analysis segment excised (so the PSD estimate is not contaminated by
    the data being analyzed).

    If `cfg.injection` is set, a simulated dark-photon signal is then added on
    top of each sensor's real segment (see `_inject_signal`); otherwise the
    real segments are left untouched, as in a plain search/PE run.
    """
    preset = get_quantum_sensor_preset()
    seg_start = cfg.trigger_time - cfg.duration + cfg.post_trigger_duration

    sensors: list[QuantumSensor] = []
    for name in cfg.sensors:
        qs = preset[name]
        qs.load_and_set_data_from_mat(
            str(cfg.sensor_files[name]),
            sig_key=cfg.sig_key,
            t_key=cfg.t_key,
            T2_key=cfg.T2_key,
            freq_key=cfg.freq_key,
            start_time=cfg.data_start_gps,
        )
        full_data = qs.data
        fs = full_data.sampling_frequency

        n_times = int(round(cfg.duration * fs))
        idx0 = int(round((seg_start - full_data.start_time) * fs))
        segment = full_data.td[idx0 : idx0 + n_times]
        segment = segment - segment.mean()
        qs.set_data(
            Data(
                td=segment,
                delta_t=full_data.delta_t,
                start_time=seg_start,
                name=f"{qs.name}_real",
            )
        )

        psd_td = jnp.concatenate([full_data.td[:idx0], full_data.td[idx0 + n_times :]])
        psd_data = Data(
            td=psd_td,
            delta_t=full_data.delta_t,
            start_time=full_data.start_time,
            name=f"{qs.name}_psd_estimation",
        )
        target_frequencies = jnp.fft.rfftfreq(n_times, d=1.0 / fs)
        psd = psd_data.to_psd(
            window=("tukey", 0.2), nperseg=int(cfg.psd_nperseg_duration * fs)
        ).interpolate(target_frequencies)
        qs.set_psd(psd)

        logger.info(
            "%s: freq_Xe=%.4f Hz, tau_Xe=%.4f s, segment %.1f s @ %.0f Hz",
            qs.name,
            qs.freq_Xe,
            qs.tau_Xe,
            cfg.duration,
            fs,
        )
        sensors.append(qs)

    if cfg.injection is not None:
        params = _resolve_injection_parameters(
            sensors, cfg.injection, waveform, f_min, f_max, cfg.trigger_time
        )
        optimal_snrs = [
            _inject_signal(qs, params, waveform, f_min, f_max, cfg.trigger_time)
            for qs in sensors
        ]
        network_snr = sum(snr**2 for snr in optimal_snrs) ** 0.5
        logger.warning(
            "Simulated injection complete -- network optimal SNR = %.2f",
            network_snr,
        )

    return sensors


def _resolve_injection_parameters(
    sensors: list[QuantumSensor],
    injection: InjectionConfig,
    waveform,
    f_min: float,
    f_max: float,
    trigger_time: float,
) -> dict[str, float]:
    """Return likelihood-space injection parameters with `d_L` resolved.

    If the user gave `d_L` directly, it is used as-is. If they gave
    `target_optimal_snr` instead, `d_L` is solved for by scaling a trial
    injection at `d_L=1` (optimal SNR scales as `1/d_L`, so one trial per
    sensor is enough) so the network optimal SNR hits the target.
    """
    params = dict(injection.injection_parameters)
    if injection.target_optimal_snr is None:
        return params

    params["d_L"] = 1.0
    network_snr_sq = sum(
        _trial_optimal_snr(qs, waveform, params, f_min, f_max, trigger_time) ** 2
        for qs in sensors
    )
    params["d_L"] = network_snr_sq**0.5 / injection.target_optimal_snr
    logger.warning(
        "Calibrated d_L = %.2f Mpc for target network optimal SNR = %.2f",
        params["d_L"],
        injection.target_optimal_snr,
    )
    return params


def _trial_optimal_snr(
    qs: QuantumSensor,
    waveform,
    params: dict[str, float],
    f_min: float,
    f_max: float,
    trigger_time: float,
) -> float:
    """Optimal SNR for `params` against qs's PSD -- independent of qs.data."""
    p = dict(params)
    p["trigger_time"] = float(trigger_time)
    p["gmst"] = float(compute_gmst(trigger_time))
    qs.set_frequency_bounds(f_min, f_max)
    band_frequencies = qs.sliced_frequencies
    band_signal = qs.fd_response(band_frequencies, waveform(band_frequencies, p), p)
    df = band_frequencies[1] - band_frequencies[0]
    return float(inner_product(band_signal, band_signal, qs.sliced_psd, df) ** 0.5)


def _inject_signal(
    qs: QuantumSensor,
    params: dict[str, float],
    waveform,
    f_min: float,
    f_max: float,
    trigger_time: float,
) -> float:
    """Add a simulated signal on top of qs's already-loaded real segment.

    Unlike `QuantumSensor.inject_signal`, which always starts from a zeroed
    data buffer, this adds the signal directly onto the existing (real)
    frequency-domain strain, so the real noise is preserved. Returns the
    injected signal's optimal SNR.
    """
    logger.warning(
        "SIMULATED SIGNAL INJECTED into %s -- this is NOT a real-data result.",
        qs.name,
    )
    p = dict(params)
    p["trigger_time"] = float(trigger_time)
    p["gmst"] = float(compute_gmst(trigger_time))

    qs.set_frequency_bounds(f_min, f_max)
    band_frequencies = qs.sliced_frequencies
    polarisations = waveform(band_frequencies, p)
    band_signal = qs.fd_response(band_frequencies, polarisations, p)

    real_fd = qs.data.fft()
    strain_data = real_fd.at[jnp.nonzero(qs.frequency_mask)[0]].add(band_signal)
    qs.set_data(
        Data.from_fd(
            fd_strain=strain_data,
            frequencies=qs.frequencies,
            start_time=qs.data.start_time,
            name=f"{qs.name}_injected",
        )
    )
    qs.set_frequency_bounds()  # reset to full range; likelihood build re-narrows it

    df = qs.sliced_frequencies[1] - qs.sliced_frequencies[0]
    optimal_snr = inner_product(band_signal, band_signal, qs.sliced_psd, df) ** 0.5
    match_filtered_snr = (
        complex_inner_product(band_signal, qs.sliced_fd_data, qs.sliced_psd, df)
        / optimal_snr
    )
    qs.optimal_snr = optimal_snr
    qs.match_filtered_snr = match_filtered_snr
    logger.info(
        "%s: injected optimal SNR = %.2f, matched-filter SNR = %s",
        qs.name,
        float(optimal_snr),
        complex(match_filtered_snr),
    )
    return float(optimal_snr)
