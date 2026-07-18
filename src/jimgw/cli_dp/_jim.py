import logging
from typing import Sequence

from jimgw.cli_dp._transforms import infer_periodic
from jimgw.core.jim import Jim
from jimgw.core.transforms import NtoMTransform

logger = logging.getLogger(__name__)

_CLI_CHECKPOINT_INTERVAL = 600.0  # 10 minutes


def _with_checkpoint(sampler_config, output_dir):
    """Return a copy of *sampler_config* with CLI checkpoint defaults applied.

    Only fields the user did not explicitly set are filled in. Mirrors
    `jimgw.cli._jim._with_checkpoint`.
    """
    explicitly_set = sampler_config.model_fields_set
    update = {}
    if "checkpoint_dir" not in explicitly_set:
        update["checkpoint_dir"] = output_dir
    if "checkpoint_interval" not in explicitly_set:
        update["checkpoint_interval"] = _CLI_CHECKPOINT_INTERVAL
    if not update:
        return sampler_config
    merged = sampler_config.model_dump() | update
    return sampler_config.__class__.model_validate(merged)


def build_jim(
    likelihood,
    prior,
    likelihood_transforms: Sequence[NtoMTransform],
    cfg,
    nf_prior=None,
    verbose: bool = False,
) -> Jim:
    """Wire together Jim from the fully-built dark-photon components.

    No sample transforms are used (see `_transforms.py`); `ra`/`phase_c`
    periodicity is passed to `Jim` directly, whether those angles come from the
    `[prior]` section or from `nf_prior`.
    """
    sampler_config = _with_checkpoint(cfg.sampler, cfg.output.dir)
    periodic = infer_periodic(cfg.prior, nf_prior)
    jim = Jim(
        likelihood=likelihood,
        prior=prior,
        sampler_config=sampler_config,
        likelihood_transforms=likelihood_transforms,
        periodic=periodic or None,
        seed=cfg.seed,
        verbose=verbose,
    )
    logger.info(
        "Built Jim (sampler=%s, seed=%d, periodic=%s)",
        cfg.sampler.type,
        cfg.seed,
        list(periodic),
    )
    return jim
