"""Unit tests for dark-photon transform and periodic-parameter inference."""

import jax.numpy as jnp
import pytest

from jimgw.cli._config import PriorConfig, UniformSpec, WaveformConfig
from jimgw.cli_dp._transforms import infer_likelihood_transforms, infer_periodic

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


_DARK_PHOTON_CFG = WaveformConfig(
    approximant="DarkPhotonWaveform", base_approximant="IMRPhenomD", f_ref=20.0
)


def test_no_likelihood_transforms_without_q_or_lambda_ratio():
    assert infer_likelihood_transforms(frozenset({"M_c"}), _DARK_PHOTON_CFG) == []


def test_likelihood_transforms_cover_q_and_lambda_ratio():
    transforms = infer_likelihood_transforms(
        frozenset({"q", "Lambda_ratio"}), _DARK_PHOTON_CFG
    )
    names = [
        t.__name__ if hasattr(t, "__name__") else type(t).__name__ for t in transforms
    ]
    assert names == ["BijectiveTransform", "ScaleTransform"]

    p = {"M_c": 30.0, "q": 0.8, "Lambda_ratio": 1.0}
    for transform in transforms:
        p = dict(transform.forward(p))
    assert p["Lambda"] == pytest.approx(2.14e4)
    assert "eta" in p
