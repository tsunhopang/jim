"""Unit tests for dark-photon periodic-parameter inference."""

import jax.numpy as jnp
import pytest

from jimgw.cli._config import PriorConfig, UniformSpec
from jimgw.cli_dp._transforms import infer_periodic

TWO_PI = 2 * jnp.pi


class StubNFPrior:
    """Stands in for a loaded `NormalizingFlowPrior`, which needs a real flow file."""

    def __init__(self, parameter_names, bounds=None):
        bounds = bounds or {}
        self.parameter_names = parameter_names
        self.lower = jnp.array(
            [bounds.get(n, (-jnp.inf, jnp.inf))[0] for n in parameter_names]
        )
        self.upper = jnp.array(
            [bounds.get(n, (-jnp.inf, jnp.inf))[1] for n in parameter_names]
        )


def test_periodic_from_prior_section():
    cfg = PriorConfig({"ra": UniformSpec(min=0.0, max=TWO_PI)})
    assert infer_periodic(cfg) == {"ra": (0.0, TWO_PI)}


def test_periodic_from_unbounded_nf_prior():
    cfg = PriorConfig({})
    nf = StubNFPrior(["M_c", "ra", "dec"])
    assert infer_periodic(cfg, nf) == {"ra": (0.0, TWO_PI)}


def test_periodic_from_nf_prior_bounded_to_full_period():
    cfg = PriorConfig({})
    nf = StubNFPrior(["ra"], bounds={"ra": (0.0, TWO_PI)})
    assert infer_periodic(cfg, nf) == {"ra": (0.0, TWO_PI)}


def test_narrow_nf_bounds_are_not_periodic(caplog):
    cfg = PriorConfig({})
    nf = StubNFPrior(["ra"], bounds={"ra": (1.0, 2.0)})
    with caplog.at_level("WARNING"):
        assert infer_periodic(cfg, nf) == {}
    assert "non-periodic" in caplog.text


def test_nf_prior_without_angles_adds_nothing():
    cfg = PriorConfig({"phase_c": UniformSpec(min=0.0, max=TWO_PI)})
    nf = StubNFPrior(["M_c", "q", "d_L"])
    assert infer_periodic(cfg, nf) == {"phase_c": (0.0, TWO_PI)}


@pytest.mark.parametrize("name", ["ra", "phase_c"])
def test_both_angles_handled_from_nf(name):
    nf = StubNFPrior([name])
    assert infer_periodic(PriorConfig({}), nf) == {name: (0.0, TWO_PI)}
