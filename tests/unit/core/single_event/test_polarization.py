"""Unit tests for wave polarization utilities."""

import jax.numpy as jnp
import pytest

from jimgw.core.single_event.polarization import (
    KNOWN_POLS,
    Polarization,
    rotated_wave_basis,
)
from tests.utils import assert_all_finite


class TestRotatedWaveBasis:
    """Test suite for the rotated_wave_basis helper."""

    def test_shape_and_finite(self):
        m, n = rotated_wave_basis(ra=1.5, dec=0.5, psi=0.3, gmst=2.0)
        assert m.shape == (3,)
        assert n.shape == (3,)
        assert_all_finite(m)
        assert_all_finite(n)

    def test_orthonormal(self):
        m, n = rotated_wave_basis(ra=1.5, dec=0.5, psi=0.3, gmst=2.0)
        assert jnp.allclose(jnp.linalg.norm(m), 1.0)
        assert jnp.allclose(jnp.linalg.norm(n), 1.0)
        assert jnp.allclose(jnp.dot(m, n), 0.0, atol=1e-7)

    def test_matches_tensor_from_sky(self):
        """m, n should reproduce exactly what tensor_from_sky uses internally."""
        ra, dec, psi, gmst = 1.5, 0.5, 0.3, 2.0
        m, n = rotated_wave_basis(ra, dec, psi, gmst)

        pol = Polarization("p")
        expected = pol.tensor_from_basis(m, n)
        actual = pol.tensor_from_sky(ra, dec, psi, gmst)
        assert jnp.allclose(actual, expected)

    def test_psi_rotation_changes_basis(self):
        m0, n0 = rotated_wave_basis(ra=1.5, dec=0.5, psi=0.0, gmst=2.0)
        m1, n1 = rotated_wave_basis(ra=1.5, dec=0.5, psi=jnp.pi / 4, gmst=2.0)
        assert not jnp.allclose(m0, m1)
        assert not jnp.allclose(n0, n1)


class TestPolarization:
    """Test suite for Polarization class."""

    def test_initialization_valid_modes(self):
        """Test initialization with valid polarization modes."""
        for mode in KNOWN_POLS:
            pol = Polarization(mode)
            assert pol.name == mode.lower()

    def test_initialization_uppercase(self):
        """Test that uppercase mode names are handled."""
        pol = Polarization("P")
        assert pol.name == "p"

        pol = Polarization("C")
        assert pol.name == "c"

    def test_initialization_invalid_mode_raises_error(self):
        """Test that invalid polarization mode raises ValueError."""
        with pytest.raises(ValueError, match="unknown mode"):
            Polarization("invalid")

        with pytest.raises(ValueError, match="unknown mode"):
            Polarization("z")

    def test_tensor_from_basis_plus(self):
        """Test plus polarization tensor generation."""
        pol = Polarization("p")

        # Define orthonormal basis vectors
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        tensor = pol.tensor_from_basis(x, y)

        # Plus polarization: x⊗x - y⊗y
        expected = jnp.outer(x, x) - jnp.outer(y, y)
        assert jnp.allclose(tensor, expected)
        assert_all_finite(tensor)

    def test_tensor_from_basis_cross(self):
        """Test cross polarization tensor generation."""
        pol = Polarization("c")

        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        tensor = pol.tensor_from_basis(x, y)

        # Cross polarization: x⊗y + y⊗x
        expected = jnp.outer(x, y) + jnp.outer(y, x)
        assert jnp.allclose(tensor, expected)
        assert_all_finite(tensor)

    def test_tensor_from_basis_vector_x(self):
        """Test vector x polarization tensor generation."""
        pol = Polarization("x")

        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        tensor = pol.tensor_from_basis(x, y)

        # Should be non-zero and finite
        assert jnp.any(jnp.abs(tensor) > 0)
        assert_all_finite(tensor)

    def test_tensor_from_basis_vector_y(self):
        """Test vector y polarization tensor generation."""
        pol = Polarization("y")

        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        tensor = pol.tensor_from_basis(x, y)

        # Should be non-zero and finite
        assert jnp.any(jnp.abs(tensor) > 0)
        assert_all_finite(tensor)

    def test_tensor_from_basis_breathing(self):
        """Test breathing polarization tensor generation."""
        pol = Polarization("b")

        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        tensor = pol.tensor_from_basis(x, y)

        # Breathing polarization: x⊗x + y⊗y
        expected = jnp.outer(x, x) + jnp.outer(y, y)
        assert jnp.allclose(tensor, expected)
        assert_all_finite(tensor)

    def test_tensor_from_basis_longitudinal(self):
        """Test longitudinal polarization tensor generation."""
        pol = Polarization("l")

        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        tensor = pol.tensor_from_basis(x, y)

        # Should be non-zero and finite
        assert jnp.any(jnp.abs(tensor) > 0)
        assert_all_finite(tensor)

    def test_tensor_from_sky_plus(self):
        """Test plus polarization tensor from sky coordinates."""
        pol = Polarization("p")

        # Example sky coordinates
        ra = 1.5  # Right ascension
        dec = 0.5  # Declination
        psi = 0.3  # Polarization angle
        gmst = 2.0  # Greenwich Mean Sidereal Time

        tensor = pol.tensor_from_sky(ra, dec, psi, gmst)

        # Should be 3x3 matrix
        assert tensor.shape == (3, 3)
        assert_all_finite(tensor)

    def test_tensor_from_sky_cross(self):
        """Test cross polarization tensor from sky coordinates."""
        pol = Polarization("c")

        ra = 1.5
        dec = 0.5
        psi = 0.3
        gmst = 2.0

        tensor = pol.tensor_from_sky(ra, dec, psi, gmst)

        assert tensor.shape == (3, 3)
        assert_all_finite(tensor)

    def test_all_polarizations_from_sky(self):
        """Test that all polarization modes can generate tensors from sky coordinates."""
        ra = 1.5
        dec = 0.5
        psi = 0.3
        gmst = 2.0

        for mode in KNOWN_POLS:
            pol = Polarization(mode)
            tensor = pol.tensor_from_sky(ra, dec, psi, gmst)

            assert tensor.shape == (3, 3)
            assert_all_finite(tensor)
            # Tensor should be non-trivial
            assert jnp.any(jnp.abs(tensor) > 0)
