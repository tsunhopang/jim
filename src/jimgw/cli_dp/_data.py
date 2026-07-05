import logging

import jax.numpy as jnp

from jimgw.cli_dp._config import DataConfig
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import QuantumSensor, get_quantum_sensor_preset

logger = logging.getLogger(__name__)


def build_sensors(cfg: DataConfig) -> list[QuantumSensor]:
    """Load real analysis segments and PSDs for the configured quantum sensors.

    For each sensor: load the full raw .mat time series (also sets
    `tau_Xe`/`freq_Xe`), slice out and demean the real analysis segment around
    the trigger time, then estimate the PSD via Welch on the full series with
    that analysis segment excised (so the PSD estimate is not contaminated by
    the data being analyzed). No signal is injected.
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

        psd_td = jnp.concatenate(
            [full_data.td[:idx0], full_data.td[idx0 + n_times :]]
        )
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

    return sensors
