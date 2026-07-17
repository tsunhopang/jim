"""``jim-nf``: fit a normalizing flow to GWTC posterior samples.

Reads posterior samples directly from a GWTC combined-PE-release HDF5 file
(pesummary layout), fits a flowjax masked-autoregressive spline flow to a chosen
set of parameters, and serializes the trained flow (plus a JSON metadata sidecar)
for later reuse as an informed prior.

Heavy imports (JAX, flowjax, equinox, h5py, corner) are deferred until inside the
command body so ``jim-nf --help`` stays fast, matching the ``jim-dp-run`` pattern.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import typer

from jimgw._logging import LOG_FORMAT

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="jim-nf",
    add_completion=False,
    help="Fit a normalizing flow to GWTC posterior samples from a PE-release HDF5 file.",
)

_DEFAULT_PARAMETERS = "chirp_mass,mass_ratio,luminosity_distance,iota,ra,dec"


def _event_from_filename(path: Path) -> str:
    """Extract the ``GW......`` event token from a GWTC release filename."""
    m = re.search(r"(GW\d{6}_\d{6})", path.name)
    return m.group(1) if m else "nf"


def _resolve_jim_names(
    source_fields: list[str], jim_parameters: Optional[str]
) -> list[str]:
    """Map read pesummary fields to jim parameter names for the metadata.

    Uses ``--jim-parameters`` verbatim when given (must match the number of fields);
    otherwise applies the built-in ``PESUMMARY_TO_JIM`` map, exiting with code 2 and
    listing any unmapped field.
    """
    # Import here (not at module top) to keep `jim-nf --help` free of the JAX-pulling
    # nf_prior module; the mapping dict itself is JAX-free but lives beside the prior.
    from jimgw.core.nf_prior import PESUMMARY_TO_JIM

    if jim_parameters is not None:
        names = [n.strip() for n in jim_parameters.split(",") if n.strip()]
        if len(names) != len(source_fields):
            typer.echo(
                f"Error: --jim-parameters has {len(names)} name(s) but --parameters "
                f"has {len(source_fields)} field(s); they must match in order.",
                err=True,
            )
            raise typer.Exit(code=2)
        return names

    unmapped = [f for f in source_fields if f not in PESUMMARY_TO_JIM]
    if unmapped:
        typer.echo(
            f"Error: no built-in jim-name mapping for {unmapped}. "
            f"Pass --jim-parameters to specify names explicitly. "
            f"Mappable fields: {sorted(PESUMMARY_TO_JIM)}",
            err=True,
        )
        raise typer.Exit(code=2)
    return [PESUMMARY_TO_JIM[f] for f in source_fields]


def _load_samples(input_path: Path, group: str, parameters: list[str]):
    """Return an ``(N, D)`` float64 array of the requested posterior fields.

    Exits with code 2 and a message listing the valid names if the group or any
    requested field is absent.
    """
    import h5py
    import numpy as np

    with h5py.File(input_path, "r") as h:
        if group not in h:
            available = [k for k in h.keys() if k not in ("history", "version")]
            typer.echo(
                f"Error: group '{group}' not found in {input_path.name}. "
                f"Available groups: {available}",
                err=True,
            )
            raise typer.Exit(code=2)
        grp = h[group]
        if not isinstance(grp, h5py.Group) or "posterior_samples" not in grp:
            typer.echo(
                f"Error: group '{group}' has no 'posterior_samples' dataset.", err=True
            )
            raise typer.Exit(code=2)
        ps = grp["posterior_samples"]
        assert isinstance(ps, h5py.Dataset)
        fields = ps.dtype.names or ()
        missing = [p for p in parameters if p not in fields]
        if missing:
            typer.echo(
                f"Error: parameter(s) {missing} not found in "
                f"{group}/posterior_samples. Available fields: {sorted(fields)}",
                err=True,
            )
            raise typer.Exit(code=2)
        cols = [np.asarray(ps[p], dtype=np.float64) for p in parameters]
    return np.stack(cols, axis=1)


@app.command()
def run(
    input: Path = typer.Option(
        ..., "--input", help="Path to the GWTC combined-PE-release HDF5 file."
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output directory for the trained model and plots."
    ),
    label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Model label. Defaults to the event token parsed from the filename.",
    ),
    group: str = typer.Option(
        "C00:Mixed", "--group", help="posterior_samples analysis group to read."
    ),
    parameters: str = typer.Option(
        _DEFAULT_PARAMETERS,
        "--parameters",
        help="Comma-separated pesummary field names to train on.",
    ),
    jim_parameters: Optional[str] = typer.Option(
        None,
        "--jim-parameters",
        help="Comma-separated jim parameter names (same order as --parameters) to "
        "store in the metadata. Overrides the built-in pesummary->jim mapping.",
    ),
    seed: int = typer.Option(42, "--seed", help="PRNG seed for flow init/training."),
    n_samples: Optional[int] = typer.Option(
        None, "--n-samples", help="Cap on the number of training samples."
    ),
    learning_rate: float = typer.Option(5e-4, "--learning-rate"),
    max_epochs: int = typer.Option(1000, "--max-epochs"),
    max_patience: int = typer.Option(30, "--max-patience"),
    knots: int = typer.Option(10, "--knots", help="RationalQuadraticSpline knots."),
    interval: float = typer.Option(
        5.0, "--interval", help="RationalQuadraticSpline interval."
    ),
    save_corner: bool = typer.Option(
        True, "--save-corner/--no-save-corner", help="Reload-and-validate corner plot."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Fit a normalizing flow to posterior samples and serialize it."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("jimgw").setLevel(level)

    if not input.exists():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(code=2)

    param_list = [p.strip() for p in parameters.split(",") if p.strip()]
    if not param_list:
        typer.echo("Error: --parameters is empty.", err=True)
        raise typer.Exit(code=2)

    jim_names = _resolve_jim_names(param_list, jim_parameters)

    model_label = label or _event_from_filename(input)
    event = _event_from_filename(input)

    import equinox as eqx
    import jax

    jax.config.update("jax_enable_x64", True)

    import numpy as np
    from flowjax.bijections import Affine, Invert
    from flowjax.distributions import Transformed
    from flowjax.train import fit_to_data

    from jimgw.core.nf_prior import (
        FLOW_LAYERS,
        NN_DEPTH,
        NN_WIDTH,
        build_flow_skeleton,
        build_inner_flow,
    )

    flow_key, train_key, sample_key = jax.random.split(jax.random.key(seed), 3)

    logger.info("Loading %s samples from %s [%s]", param_list, input.name, group)
    x = _load_samples(input, group, param_list)
    n_dim = x.shape[1]
    if n_samples is not None:
        x = x[:n_samples]
    logger.info("Training on %d samples, %d parameters", x.shape[0], n_dim)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    preprocess = Affine(-mean / std, 1.0 / std)
    x_processed = jax.vmap(preprocess.transform)(x)

    flow = build_inner_flow(flow_key, n_dim, knots, interval)
    flow, losses = fit_to_data(
        key=train_key,
        dist=flow,
        data=x_processed,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
    )
    flow = Transformed(flow, Invert(preprocess))
    logger.info("Training complete. Final val loss: %.4f", float(losses["val"][-1]))

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{model_label}_NF.eqx"
    eqx.tree_serialise_leaves(str(model_path), flow)

    meta = {
        "parameters": jim_names,
        "source_fields": param_list,
        "group": group,
        "event": event,
        "seed": seed,
        "knots": knots,
        "interval": interval,
        "n_dim": n_dim,
        "flow_layers": FLOW_LAYERS,
        "nn_width": NN_WIDTH,
        "nn_depth": NN_DEPTH,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    meta_path = output / f"{model_label}_NF.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved flow to %s and metadata to %s", model_path, meta_path)

    if save_corner:
        clean_flow = build_flow_skeleton(jax.random.key(seed), n_dim, knots, interval)
        loaded_flow = eqx.tree_deserialise_leaves(str(model_path), clean_flow)
        nf_samples = np.asarray(loaded_flow.sample(sample_key, (x.shape[0],)))
        corner_path = output / f"{model_label}_reloaded_corner.png"
        _make_cornerplot(x, nf_samples, jim_names, str(corner_path))
        logger.info("Saved validation corner plot to %s", corner_path)


def _make_cornerplot(train_data, nf_samples, labels: list[str], name: str) -> None:
    """Overlay training data (blue) and reloaded-flow samples (red)."""
    import corner
    import matplotlib.pyplot as plt
    import numpy as np

    parameter_range = [
        (float(np.min(train_data[:, i])), float(np.max(train_data[:, i])))
        for i in range(train_data.shape[1])
    ]
    kwargs: dict[str, Any] = dict(
        bins=40,
        smooth=1.0,
        labels=labels,
        show_titles=False,
        levels=[0.68, 0.95, 0.997],
        plot_density=False,
        plot_datapoints=False,
        fill_contours=False,
        max_n_ticks=4,
        min_n_ticks=3,
        density=True,
        range=parameter_range,
    )
    fig = corner.corner(
        train_data,
        color="blue",
        hist_kwargs={"density": True, "color": "blue"},
        **kwargs,
    )
    corner.corner(
        nf_samples,
        fig=fig,
        color="red",
        hist_kwargs={"density": True, "color": "red"},
        **kwargs,
    )
    plt.text(
        0.75,
        0.75,
        "Training data",
        fontsize=24,
        color="blue",
        transform=fig.transFigure,
    )
    plt.text(
        0.75,
        0.68,
        "Normalizing flow",
        fontsize=24,
        color="red",
        transform=fig.transFigure,
    )
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)
