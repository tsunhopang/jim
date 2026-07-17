import logging
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from jimgw._logging import LOG_FORMAT
from jimgw.cli_dp._config import PipelineConfig

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="jim-dp-run",
    add_completion=False,
    help="Run a jimgw dark-photon parameter-estimation pipeline from a TOML config file.",
)

_INIT_TEMPLATE = """\
seed = 0

[data]
sensors = ["QS-I", "QS-II", "QS-III", "QS-IV", "QS-V"]
sensor_files = { "QS-I" = "real_data/Sensor1.mat", "QS-II" = "real_data/Sensor2.mat", "QS-III" = "real_data/Sensor3.mat", "QS-IV" = "real_data/Sensor4.mat", "QS-V" = "real_data/Sensor5.mat" }
data_start_gps = 1343650218.0
trigger_time = 1343651418.0
duration = 8.0
post_trigger_duration = 2.0
psd_nperseg_duration = 8.0

# To validate the pipeline by injecting a simulated signal into the real
# segment, uncomment and adjust (off by default). Give exactly one of
# injection_parameters.d_L or target_optimal_snr:
# [data.injection]
# target_optimal_snr = 30.0
# injection_parameters = { M_c = 30.0, eta = 0.25, s1_z = 0.0, s2_z = 0.0, t_c = 0.0, phase_c = 0.0, iota = 0.0, ra = 1.5, dec = 0.5, psi = 0.3, sigma_1 = 0.2, sigma_2 = -0.2, eps_BD = 0.5 }

[waveform]
approximant = "DarkPhotonWaveform"
base_approximant = "IMRPhenomD"
f_ref = 20.0

[prior]
M_c     = { type = "uniform",   min = 25.0,  max = 35.0  }
q       = { type = "uniform",   min = 0.125, max = 1.0   }
s1_z    = { type = "uniform",   min = -0.99, max = 0.99  }
s2_z    = { type = "uniform",   min = -0.99, max = 0.99  }
iota    = { type = "sine" }
d_L     = { type = "power_law", min = 1.0,   max = 2000.0, alpha = 2.0 }
t_c     = { type = "uniform",   min = -0.1,  max = 0.1   }
phase_c = { type = "uniform",   min = 0.0,   max = 6.283185307179586 }  # 2π
psi     = { type = "uniform",   min = 0.0,   max = 3.141592653589793 }  # π
ra      = { type = "uniform",   min = 0.0,   max = 6.283185307179586 }  # 2π
dec     = { type = "cosine" }
sigma_1 = { type = "uniform",   min = -0.5,  max = 0.5   }
sigma_2 = { type = "uniform",   min = -0.5,  max = 0.5   }
eps_BD  = { type = "uniform",   min = 0.0,   max = 1.0   }

# To use a normalizing flow trained with `jim-nf` as an informed joint prior over
# the parameters it was trained on, uncomment below and REMOVE those parameters from
# [prior] above (they are supplied by the flow). Parameter names come from the flow
# metadata. Optionally restrict each to a physical box via [nf_prior.bounds].
# [nf_prior]
# model = "output/nf/GW230605_065343_NF.eqx"
# [nf_prior.bounds]
# q   = [0.125, 1.0]
# d_L = [1.0, 2000.0]

[likelihood]
f_min = 20.0
f_max = 512.0

# Production defaults — for a quick test try: n_particles=500, n_mcmc_steps_per_dim=20
[sampler]
type = "blackjax-smc"
n_particles = 5000
n_mcmc_steps_per_dim = 100
target_ess = 10000
initial_cov_scale = 0.5
target_acceptance_rate = 0.234
scale_adaptation_gain = 3.0
persistent_sampling = true

[output]
dir = "output/my_dp_run"
# save_corner requires the 'corner' package: pip install corner
save_corner = false
"""


@app.command()
def run(
    config: Optional[Path] = typer.Argument(None, help="Path to the TOML config file."),
    init: Optional[Path] = typer.Option(
        None,
        "--init",
        help="Write a minimal quantum-sensor template config to PATH and exit.",
        metavar="PATH",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
) -> None:
    """Run a jimgw dark-photon parameter-estimation pipeline from CONFIG."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # jimgw logger is isolated (propagate=False), so set its level directly.
    logging.getLogger("jimgw").setLevel(level)

    if init is not None:
        if init.exists():
            typer.echo(
                f"Error: {init} already exists. Choose a different path.", err=True
            )
            raise typer.Exit(code=2)
        try:
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text(_INIT_TEMPLATE)
        except OSError as exc:
            typer.echo(f"Error: could not write template to {init}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"Template config written to {init}")
        raise typer.Exit()

    if config is None:
        typer.echo(
            "Error: provide a CONFIG file or use --init to create a template.", err=True
        )
        raise typer.Exit(code=2)

    if not config.exists():
        typer.echo(f"Error: config file not found: {config}", err=True)
        raise typer.Exit(code=2)

    try:
        with open(config, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        typer.echo(f"Error reading config {config}:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc

    logger.info("Loaded config from %s", config)

    try:
        cfg = PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        typer.echo(f"Config validation error:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc

    _log_config_summary(cfg)
    _log_versions(cfg.sampler.type)

    out_dir = cfg.output.dir
    if out_dir.exists() and not cfg.output.overwrite:
        typer.echo(
            f"Error: output directory already exists: {out_dir}. "
            "Set output.overwrite = true to allow overwriting.",
            err=True,
        )
        raise typer.Exit(code=2)

    import jax

    jax.config.update("jax_enable_x64", True)

    from jimgw.cli._output import write_outputs
    from jimgw.cli._prior import build_prior
    from jimgw.cli._waveform import build_waveform
    from jimgw.cli_dp._data import build_sensors
    from jimgw.cli_dp._jim import build_jim
    from jimgw.cli_dp._likelihood import build_likelihood
    from jimgw.cli_dp._nf_prior import build_nf_prior
    from jimgw.cli_dp._transforms import infer_likelihood_transforms
    from jimgw.core.prior import CombinePrior

    trigger_time: float = cfg.data.trigger_time

    # Stage 2: waveform
    waveform = build_waveform(cfg.waveform)

    # Stage 3: data — real segment + PSD per sensor, optional injection
    sensors = build_sensors(
        cfg.data, waveform, cfg.likelihood.f_min, cfg.likelihood.f_max
    )

    # Stage 4: prior — config priors, optionally joined with a trained NF prior
    prior = build_prior(cfg.prior)
    if cfg.nf_prior is not None:
        nf_prior = build_nf_prior(cfg.nf_prior)
        overlap = set(nf_prior.parameter_names) & set(cfg.prior.root.keys())
        if overlap:
            typer.echo(
                f"Error: parameter(s) {sorted(overlap)} are provided by both the NF "
                "prior and the [prior] section. Remove them from [prior].",
                err=True,
            )
            raise typer.Exit(code=2)
        prior = CombinePrior([nf_prior, prior])

    # Stage 5: likelihood transform inference (only q -> eta, if present)
    prior_params = frozenset(prior.parameter_names)
    likelihood_transforms = infer_likelihood_transforms(prior_params)

    # Stage 6: likelihood
    likelihood = build_likelihood(cfg.likelihood, sensors, waveform, trigger_time)

    # Stage 7: build Jim + run sampler
    jim = build_jim(likelihood, prior, likelihood_transforms, cfg, verbose=verbose)

    try:
        jim.sample()
    except Exception as exc:
        logger.error("Sampling failed: %s", exc)
        raise typer.Exit(code=3) from exc
    logger.info("Sampling complete.")

    # Stage 8: write outputs
    try:
        write_outputs(jim, cfg)
    except FileExistsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _log_versions(sampler_type: str) -> None:
    dists = ["JimGW", "rippleGW"]
    if sampler_type == "flowmc":
        dists.append("flowMC")
    parts = []
    for dist in dists:
        try:
            parts.append(f"{dist} {version(dist)}")
        except PackageNotFoundError:
            pass
    if parts:
        logger.info(" | ".join(parts))


def _log_config_summary(cfg: PipelineConfig) -> None:
    logger.info("seed: %d", cfg.seed)
    logger.info("data: sensors=%s", cfg.data.sensors)
    if cfg.data.injection is not None:
        logger.warning(
            "SIMULATED INJECTION ENABLED: a synthetic signal will be added "
            "to the real data segment before sampling."
        )
    logger.info(
        "waveform: %s (base=%s, f_ref=%.1f Hz)",
        cfg.waveform.approximant,
        cfg.waveform.base_approximant,
        cfg.waveform.f_ref,
    )
    param_names = list(cfg.prior.root.keys())
    logger.info("prior: %d parameter(s): %s", len(param_names), param_names)
    if cfg.nf_prior is not None:
        logger.info("nf_prior: %s", cfg.nf_prior.model)
    logger.info("sampler: type=%s", cfg.sampler.type)
    logger.info("output: %s", cfg.output.dir)
