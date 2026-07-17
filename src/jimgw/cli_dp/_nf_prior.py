"""Build a `NormalizingFlowPrior` from the dark-photon CLI config."""

import logging

from jimgw.cli_dp._config import NFPriorConfig

logger = logging.getLogger(__name__)


def build_nf_prior(cfg: NFPriorConfig):
    """Load the trained flow and wrap it as a `NormalizingFlowPrior`.

    Parameter names come from the flow metadata (already jim names, converted at
    training time), so no name mapping happens here.
    """
    # Deferred so importing the CLI config module stays JAX-free.
    from jimgw.core.nf_prior import load_nf_prior

    if not cfg.model.exists():
        raise FileNotFoundError(f"NF model file not found: {cfg.model}")
    metadata = (
        cfg.metadata if cfg.metadata is not None else cfg.model.with_suffix(".json")
    )
    if not metadata.exists():
        raise FileNotFoundError(f"NF metadata file not found: {metadata}")

    prior = load_nf_prior(cfg.model, metadata, cfg.bounds or None)

    unknown_bounds = set(cfg.bounds) - set(prior.parameter_names)
    if unknown_bounds:
        raise ValueError(
            f"nf_prior.bounds names {sorted(unknown_bounds)} are not NF parameters "
            f"{list(prior.parameter_names)}"
        )

    logger.info(
        "Built NF prior from %s: %d parameter(s): %s",
        cfg.model.name,
        len(prior.parameter_names),
        list(prior.parameter_names),
    )
    return prior
