import jax
from jax.lax import while_loop
import jax.numpy as jnp
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, Int, PRNGKeyArray, Bool, jaxtyped
from typing import Callable, Union
from dataclasses import field
from jimgw.single_event.utils import (
    EOS_check,
    detector_frame_Mc_q_to_source_frame_m1_m2,
)

from joseTOV.sampling import MetaModel_with_CSE_parameters_to_family


class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    naming: list[str]
    transforms: dict[str, tuple[str, Callable]] = field(default_factory=dict)

    @property
    def n_dim(self):
        return len(self.naming)

    def __init__(
        self, naming: list[str], transforms: dict[str, tuple[str, Callable]] = {}
    ):
        """
        Parameters
        ----------
        naming : list[str]
            A list of names for the parameters of the prior.
        transforms : dict[tuple[str,Callable]]
            A dictionary of transforms to apply to the parameters. The keys are
            the names of the parameters and the values are a tuple of the name
            of the transform and the transform itself.
        """
        self.naming = naming
        self.transforms = {}

        def make_lambda(name):
            return lambda x: x[name]

        for name in naming:
            if name in transforms:
                self.transforms[name] = transforms[name]
            else:
                # Without the function, the lambda will refer to the variable name instead of its value,
                # which will make lambda reference the last value of the variable name
                self.transforms[name] = (name, make_lambda(name))

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Apply the transforms to the parameters.

        Parameters
        ----------
        x : dict
            A dictionary of parameters. Names should match the ones in the prior.

        Returns
        -------
        x : dict
            A dictionary of parameters with the transforms applied.
        """
        output = {}
        for value in self.transforms.values():
            output[value[0]] = value[1](x)
        return output

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).
        """

        return dict(zip(self.naming, x))

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        raise NotImplementedError

    def log_prob(self, x: dict[str, Array]) -> Float:
        raise NotImplementedError


@jaxtyped
class Uniform(Prior):
    xmin: Float = 0.0
    xmax: Float = 1.0

    def __repr__(self):
        return f"Uniform(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Uniform needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.uniform(
            rng_key, (n_samples,), minval=self.xmin, maxval=self.xmax
        )
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        output = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        return output + jnp.log(1.0 / (self.xmax - self.xmin))


@jaxtyped
class OrderedUniform(Prior):
    xmin: Float = 0.0
    xmax: Float = 1.0
    descending: Bool = True

    def __repr__(self):
        return f"OrderUniform(xmin={self.xmin}, xmax={self.xmax}, n_dim={self.n_dim})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        descending: Bool,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim > 1, "OrderUniform needs to be at least 2D distribution"
        self.xmax = xmax
        self.xmin = xmin
        self.descending = descending

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from an ordered uniform distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.uniform(
            rng_key, (self.n_dim, n_samples), minval=self.xmin, maxval=self.xmax
        )
        samples = jnp.sort(samples, axis=0, descending=self.descending)
        return self.add_name(samples)

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = jnp.array([x.get(name) for name in self.naming])
        sorted_variable = jnp.sort(variable, axis=0, descending=self.descending)

        in_range_condition = (variable >= self.xmax) * (variable <= self.xmin)
        sorted_condition = jnp.all(variable == sorted_variable, axis=0)
        sorted_condition *= jnp.ones(variable.shape)

        output = jnp.where(
            in_range_condition * sorted_condition,
            jnp.zeros_like(variable),
            jnp.zeros_like(variable) - jnp.inf,
        )
        return output + self.n_dim * jnp.log(1.0 / (self.xmax - self.xmin))


@jaxtyped
class Unconstrained_Uniform(Prior):
    xmin: Float = 0.0
    xmax: Float = 1.0

    def __repr__(self):
        return f"Unconstrained_Uniform(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Unconstrained_Uniform needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        local_transform = self.transforms

        def new_transform(param):
            param[self.naming[0]] = self.to_range(param[self.naming[0]])
            return local_transform[self.naming[0]][1](param)

        self.transforms = {
            self.naming[0]: (local_transform[self.naming[0]][0], new_transform)
        }

    def to_range(self, x: Float) -> Float:
        """
        Transform the parameters to the range of the prior.

        Parameters
        ----------
        x : Float
            The parameters to transform.

        Returns
        -------
        x : dict
            A dictionary of parameters with the transforms applied.
        """
        return (self.xmax - self.xmin) / (1 + jnp.exp(-x)) + self.xmin

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples :
            An array of shape (n_samples, n_dim) containing the samples.

        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0, maxval=1)
        samples = jnp.log(samples / (1 - samples))
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        return jnp.log(jnp.exp(-variable) / (1 + jnp.exp(-variable)) ** 2)


class Sphere(Prior):
    """
    A prior on a sphere represented by Cartesian coordinates.

    Magnitude is sampled from a uniform distribution.
    """

    def __repr__(self):
        return f"Sphere(naming={self.naming})"

    def __init__(self, naming: str, **kwargs):
        self.naming = [f"{naming}_theta", f"{naming}_phi", f"{naming}_mag"]
        self.transforms = {
            self.naming[0]: (
                f"{naming}_x",
                lambda params: jnp.sin(params[self.naming[0]])
                * jnp.cos(params[self.naming[1]])
                * params[self.naming[2]],
            ),
            self.naming[1]: (
                f"{naming}_y",
                lambda params: jnp.sin(params[self.naming[0]])
                * jnp.sin(params[self.naming[1]])
                * params[self.naming[2]],
            ),
            self.naming[2]: (
                f"{naming}_z",
                lambda params: jnp.cos(params[self.naming[0]]) * params[self.naming[2]],
            ),
        }

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        rng_keys = jax.random.split(rng_key, 3)
        theta = jnp.arccos(
            jax.random.uniform(rng_keys[0], (n_samples,), minval=-1.0, maxval=1.0)
        )
        phi = jax.random.uniform(rng_keys[1], (n_samples,), minval=0, maxval=2 * jnp.pi)
        mag = jax.random.uniform(rng_keys[2], (n_samples,), minval=0, maxval=1)
        return self.add_name(jnp.stack([theta, phi, mag], axis=1).T)

    def log_prob(self, x: dict[str, Float]) -> Float:
        theta = x[self.naming[0]]
        phi = x[self.naming[1]]
        mag = x[self.naming[2]]
        output = jnp.where(
            (mag > 1)
            | (mag < 0)
            | (phi > 2 * jnp.pi)
            | (phi < 0)
            | (theta > jnp.pi)
            | (theta < 0),
            jnp.zeros_like(0) - jnp.inf,
            jnp.log(mag**2 * jnp.sin(x[self.naming[0]])),
        )
        return output


@jaxtyped
class AlignedSpin(Prior):
    """
    Prior distribution for the aligned (z) component of the spin.

    This assume the prior distribution on the spin magnitude to be uniform in [0, amax]
    with its orientation uniform on a sphere

    p(chi) = -log(|chi| / amax) / 2 / amax

    This is useful when comparing results between an aligned-spin run and
    a precessing spin run.

    See (A7) of https://arxiv.org/abs/1805.10457.
    """

    amax: Float = 0.99
    chi_axis: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))
    cdf_vals: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))

    def __repr__(self):
        return f"Alignedspin(amax={self.amax}, naming={self.naming})"

    def __init__(
        self,
        amax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Alignedspin needs to be 1D distributions"
        self.amax = amax

        # build the interpolation table for the ppf of the one-sided distribution
        chi_axis = jnp.linspace(1e-31, self.amax, num=1000)
        cdf_vals = -chi_axis * (jnp.log(chi_axis / self.amax) - 1.0) / self.amax
        self.chi_axis = chi_axis
        self.cdf_vals = cdf_vals

    @property
    def xmin(self):
        return -self.amax

    @property
    def xmax(self):
        return self.amax

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from the Alignedspin distribution.

        for chi > 0;
        p(chi) = -log(chi / amax) / amax  # halved normalization constant
        cdf(chi) = -chi * (log(chi / amax) - 1) / amax

        Since there is a pole at chi=0, we will sample with the following steps
        1. Map the samples with quantile > 0.5 to positive chi and negative otherwise
        2a. For negative chi, map the quantile back to [0, 1] via q -> 2(0.5 - q)
        2b. For positive chi, map the quantile back to [0, 1] via q -> 2(q - 0.5)
        3. Map the quantile to chi via the ppf by checking against the table
           built during the initialization
        4. add back the sign

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        # 1. calculate the sign of chi from the q_samples
        sign_samples = jnp.where(
            q_samples >= 0.5,
            jnp.zeros_like(q_samples) + 1.0,
            jnp.zeros_like(q_samples) - 1.0,
        )
        # 2. remap q_samples
        q_samples = jnp.where(
            q_samples >= 0.5,
            2 * (q_samples - 0.5),
            2 * (0.5 - q_samples),
        )
        # 3. map the quantile to chi via interpolation
        samples = jnp.interp(
            q_samples,
            self.cdf_vals,
            self.chi_axis,
        )
        # 4. add back the sign
        samples *= sign_samples

        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        log_p = jnp.where(
            (variable >= self.amax) | (variable <= -self.amax),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.log(-jnp.log(jnp.absolute(variable) / self.amax) / 2.0 / self.amax),
        )
        return log_p


@jaxtyped
class PowerLaw(Prior):
    """
    A prior following the power-law with alpha in the range [xmin, xmax).
    p(x) ~ x^{\alpha}
    """

    xmin: Float = 0.0
    xmax: Float = 1.0
    alpha: Float = 0.0
    normalization: Float = 1.0

    def __repr__(self):
        return f"Powerlaw(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.naming})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        alpha: Union[Int, Float],
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        if alpha < 0.0:
            assert xmin > 0.0, "With negative alpha, xmin must > 0"
        assert self.n_dim == 1, "Powerlaw needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha
        if alpha == -1:
            self.normalization = 1.0 / jnp.log(self.xmax / self.xmin)
        else:
            self.normalization = (1 + self.alpha) / (
                self.xmax ** (1 + self.alpha) - self.xmin ** (1 + self.alpha)
            )

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a power-law distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        if self.alpha == -1:
            samples = self.xmin * jnp.exp(q_samples * jnp.log(self.xmax / self.xmin))
        else:
            samples = (
                self.xmin ** (1.0 + self.alpha)
                + q_samples
                * (self.xmax ** (1.0 + self.alpha) - self.xmin ** (1.0 + self.alpha))
            ) ** (1.0 / (1.0 + self.alpha))
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        log_in_range = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        log_p = self.alpha * jnp.log(variable) + jnp.log(self.normalization)
        return log_p + log_in_range


@jaxtyped
class Exponential(Prior):
    """
    A prior following the power-law with alpha in the range [xmin, xmax).
    p(x) ~ exp(\alpha x)
    """

    xmin: Float = 0.0
    xmax: Float = jnp.inf
    alpha: Float = -1.0
    normalization: Float = 1.0

    def __repr__(self):
        return f"Exponential(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.naming})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        alpha: Union[Int, Float],
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        if alpha < 0.0:
            assert xmin != -jnp.inf, "With negative alpha, xmin must finite"
        if alpha > 0.0:
            assert xmax != jnp.inf, "With positive alpha, xmax must finite"
        assert not jnp.isclose(alpha, 0.0), "alpha=zero is given, use Uniform instead"
        assert self.n_dim == 1, "Exponential needs to be 1D distributions"

        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha

        self.normalization = self.alpha / (
            jnp.exp(self.alpha * self.xmax) - jnp.exp(self.alpha * self.xmin)
        )

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a exponential distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        samples = (
            self.xmin
            + jnp.log1p(
                q_samples * (jnp.exp(self.alpha * (self.xmax - self.xmin)) - 1.0)
            )
            / self.alpha
        )
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        log_in_range = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        log_p = self.alpha * variable + jnp.log(self.normalization)
        return log_p + log_in_range


class Composite(Prior):
    priors: list[Prior] = field(default_factory=list)

    def __repr__(self):
        return f"Composite(priors={self.priors}, naming={self.naming})"

    def __init__(
        self,
        priors: list[Prior],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        naming = []
        self.transforms = {}
        for prior in priors:
            naming += prior.naming
            self.transforms.update(prior.transforms)
        self.priors = priors
        self.naming = naming
        self.transforms.update(transforms)

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = {}
        for prior in self.priors:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, x: dict[str, Float]) -> Float:
        output = 0.0
        for prior in self.priors:
            output += prior.log_prob(x)
        return output


class MetaModel_CSE_EOS(Prior):

    sample_vec: Callable
    log_prob_vec: Callable
    priors: list[Prior] = field(default_factory=list)
    naming: list[str] = field(default_factory=list)
    sym_ordering: list[str] = field(default_factory=list)
    sat_ordering: list[str] = field(default_factory=list)
    n_cs_node: int = 5
    Hubble_constant: Float = 67.4
    c: Float = 299792458.0

    def __init__(
        self,
        priors: list[Prior],
        sym_ordering: list[str] = ["E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym"],
        sat_ordering: list[str] = ["E_sat", "K_sat", "Q_sat", "Z_sat"],
        n_cs_node: int = 5,
        Hubble_constant: Float = 67.4,  # km s^-1 Mpc^-1
        c: Float = 299792458.0,
    ):

        self.priors = Composite(priors)
        self.sym_ordering = sym_ordering
        self.sat_ordering = sat_ordering
        self.n_cs_node = n_cs_node
        self.Hubble_constant = Hubble_constant
        self.c = c

        self.sample_vec = jax.vmap(self.sample_single, in_axes=(0,))
        self.log_prob_vec = jax.vmap(self.log_prob_single, in_axes=(1,))

        naming = []
        for prior in priors:
            naming.extend(prior.naming)

        super().__init__(naming, {})

    def __repr__(self):
        return f"MetaModel_CSE_EOS(priors={self.priors})"

    def sample(self, rng_key: PRNGKeyArray, n_samples: int):
        key_array = jax.random.split(rng_key, n_samples)
        samples = self.sample_vec(key_array)
        samples = jnp.swapaxes(samples, 0, 1)
        return self.add_name(samples)

    def log_prob(self, x: dict[str, Array]) -> Float:
        x_unnamed = jnp.array([x[name] for name in self.naming])
        return self.log_prob_vec(x_unnamed)

    def parameters_to_EOS(self, x: dict[str, Array]) -> tuple[Array, Array, Array]:
        # put MM parameters in list
        MM_sym_params = []
        MM_sat_params = []

        for key in self.sym_ordering:
            value = x.get(key, jnp.array([0.0])).at[0].get()
            MM_sym_params.append(value)

        for key in self.sat_ordering:
            value = x.get(key, jnp.array([0.0])).at[0].get()
            MM_sat_params.append(value)

        # put CSE parameters in list
        CSE_cs2_nodes = []
        CSE_den_nodes = []
        for i in range(1, self.n_cs_node + 1):
            cs2_value = x.get(f"cs2_nodes_{i}").at[0].get()
            den_value = x.get(f"den_nodes_{i}").at[0].get()
            CSE_cs2_nodes.append(cs2_value)
            CSE_den_nodes.append(den_value)

        # get the macro-EOS
        _, masses, radii, lambdas = MetaModel_with_CSE_parameters_to_family(
            jnp.array(MM_sat_params),
            jnp.array(MM_sym_params),
            1.5 * 0.16,
            jnp.array(CSE_den_nodes),
            jnp.array(CSE_cs2_nodes),
            0.16,
            12 * 0.16,
        )

        return masses, radii, lambdas

    def sample_single(self, rng_key: PRNGKeyArray) -> Array:

        def body(rng_key: PRNGKeyArray) -> PRNGKeyArray:
            # spliting to key to create new key
            # taking the second one to avoid taking the
            # same key as in flowMC
            _, rng_key = jax.random.split(rng_key, 2)
            return rng_key

        def cond(rng_key: PRNGKeyArray) -> Bool:
            # get a trial sample
            parameters = self.priors.sample(rng_key, 1)
            # convert the masses into source frame masses
            # assume the prior is on chirp mass and mass ratio for now
            m1_source, m2_source = detector_frame_Mc_q_to_source_frame_m1_m2(
                parameters["M_c"],
                parameters["q"],
                parameters["d_L"],
                self.Hubble_constant,
                self.c,
            )

            masses, radii, lambdas = self.parameters_to_EOS(parameters)

            out = not EOS_check(m1_source, m2_source, masses, radii, lambdas)

            return jnp.all(parameters["K_sat"] >= 230.0)

        # find the key which give a valid EOS and masses
        valid_key = while_loop(
            cond_fun=cond,
            body_fun=body,
            init_val=rng_key,
        )

        # since the sample is expected to return ndarray rather than dict
        # we need to extract the values from these dictionaries
        valid_sample = self.priors.sample(valid_key, 1)

        return jnp.array([valid_sample[name] for name in self.naming])

    def log_prob_single(self, x: Array) -> Float:
        x_named = self.add_name(x)
        masses, radii, lambdas = self.parameters_to_EOS(x_named)
        m1_source, m2_source = detector_frame_Mc_q_to_source_frame_m1_m2(
            x_named["M_c"], x_named["q"], x_named["d_L"], self.Hubble_constant, self.c
        )

        output = jnp.where(
            EOS_check(
                m1_source,
                m2_source,
                masses,
                radii,
                lambdas,
            ),
            0.0,
            -jnp.inf,
        )

        return output + self.priors.log_prob(x)
