# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Von Mises distribution."""

import functools
import math
from typing import Sequence, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions import normal
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
IntLike = Union[int, np.int16, np.int32, np.int64]


class VonMises(distribution.Distribution):
  """The von Mises distribution over angles.

  The von Mises distribution is a distribution over angles. It is the maximum
  entropy distribution on the space of angles, given a circular mean and a
  circular variance.

  In this implementation, the distribution is defined over the range [-pi, pi),
  with all samples in this interval and the CDF is constant outside this
  interval. Do note that the prob and log_prob also accept values outside of
  [-pi, pi) and will return values as if they are inside the interval.

  When `concentration=0`, this distribution becomes the uniform distribution
  over the interval [-pi, pi). When the concentration goes to infinity, this
  distribution approximates a Normal distribution.

  #### Details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; loc, concentration) = exp(concentration * cos(x - loc))
  / (2 * pi * I_0 (concentration))
  ```

  where:
  * `I_0` is the zeroth order modified Bessel function;
  * `loc` the circular mean of the distribution, a scalar in radians.
    It can take arbitrary values, also outside of [-pi, pi).
  * `concentration >= 0` is the concentration parameter. It is the
    analogue to 1/sigma of the Normal distribution.

  #### Examples

  Examples of initialization of this distribution.

  ```python
  # Create a batch of two von Mises distributions.
  dist = distrax.VonMises(loc=[1.0, 2.0], concentration=[3.0, 4.0])
  dist.sample(sample_shape=(3,), seed=0)  # Sample of shape [3, 2]
  ```

  Arguments are broadcast when possible.

  ```python
  dist = distrax.VonMises(loc=1.0, concentration=[3.0, 4.0])

  # Evaluating the pdf of both distributions on the point 3.0 returns a length 2
  # tensor.
  dist.prob(3.0)
  ```
  """
  equiv_tfp_cls = tfd.VonMises

  def __init__(self, loc: Numeric, concentration: Numeric):
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._concentration = conversion.as_float_array(concentration)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._loc.shape, self._concentration.shape
    )

  @property
  def loc(self) -> Array:
    """The circular mean of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def concentration(self) -> Array:
    """The concentration of the distribution."""
    return jnp.broadcast_to(self._concentration, self.batch_shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  def mean(self) -> Array:
    """The circular mean of the distribution."""
    return self.loc

  def variance(self) -> Array:
    """The circular variance of the distribution."""
    conc = self._concentration
    return 1. - jax.scipy.special.i1e(conc) / jax.scipy.special.i0e(conc)

  def prob(self, value: Array) -> Array:
    """The probability of value under the distribution."""
    conc = self._concentration
    unnormalized_prob = jnp.exp(conc * (jnp.cos(value - self._loc) - 1.))
    normalization = (2. * math.pi) * jax.scipy.special.i0e(conc)
    return unnormalized_prob / normalization

  def log_prob(self, value: Array) -> Array:
    """The logarithm of the probability of value under the distribution."""
    conc = self._concentration
    i_0 = jax.scipy.special.i0(conc)
    return (
        conc * jnp.cos(value - self._loc) - math.log(2 * math.pi) - jnp.log(i_0)
    )

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """Returns `n` samples in [-pi, pi)."""
    out_shape = (n,) + self.batch_shape
    conc = self._concentration
    dtype = jnp.result_type(self._loc, self._concentration)
    sample = _von_mises_sample(out_shape, conc, key, dtype) + self._loc
    return _convert_angle_to_standard(sample)

  def entropy(self) -> Array:
    """Returns the entropy."""
    conc = self._concentration
    i0e = jax.scipy.special.i0e(conc)
    i1e = jax.scipy.special.i1e(conc)
    return conc * (1 - i1e / i0e) + math.log(2 * math.pi) + jnp.log(i0e)

  def mode(self) -> Array:
    """The mode of the distribution."""
    return self.mean()

  def cdf(self, value: Array) -> Array:
    """The CDF of `value` under the distribution.

    Note that the CDF takes values of 0. or 1. for values outside of
    [-pi, pi). Note that this behaviour is different from
    `tensorflow_probability.VonMises` or `scipy.stats.vonmises`.
    Args:
      value: the angle evaluated under the distribution.
    Returns:
      the circular CDF of value.
    """
    dtype = jnp.result_type(value, self._loc, self._concentration)
    loc = _convert_angle_to_standard(self._loc)
    return jnp.clip(
        _von_mises_cdf(value - loc, self._concentration, dtype)
        - _von_mises_cdf(-math.pi - loc, self._concentration, dtype),
        a_min=0.,
        a_max=1.,
    )

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def survival_function(self, value: Array) -> Array:
    """See `Distribution.survival_function`."""
    dtype = jnp.result_type(value, self._loc, self._concentration)
    loc = _convert_angle_to_standard(self._loc)
    return jnp.clip(
        _von_mises_cdf(math.pi - loc, self._concentration, dtype)
        - _von_mises_cdf(value - loc, self._concentration, dtype),
        a_min=0.,
        a_max=1.,
    )

  def log_survival_function(self, value: Array) -> Array:
    """See `Distribution.log_survival_function`."""
    return jnp.log(self.survival_function(value))

  def __getitem__(self, index) -> 'VonMises':
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return VonMises(
        loc=self.loc[index],
        concentration=self.concentration[index],
    )


def _convert_angle_to_standard(angle: Array) -> Array:
  """Converts angle in radians to representation between [-pi, pi)."""
  num_periods = jnp.round(angle / (2 * math.pi))
  rep = angle - (2 * math.pi) * num_periods
  return rep


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3))
def _von_mises_sample(
    shape: Union[IntLike, Sequence[IntLike]],
    concentration: Array,
    seed: PRNGKey,
    dtype: jnp.dtype,
) -> Array:
  """Rejection sample from the standard von Mises which has loc=0."""
  concentration = jnp.asarray(concentration, dtype=dtype)
  s_concentration_cutoff_dict = {
      jnp.float16.dtype: 1.8e-1,
      jnp.float32.dtype: 2e-2,
      jnp.float64.dtype: 1.2e-4,
  }
  s_concentration_cutoff = s_concentration_cutoff_dict[dtype]
  use_exact = concentration > s_concentration_cutoff
  # Avoid NaN's, even when not used later.
  conc = jnp.where(use_exact, concentration, 1.)
  r = 1. + jnp.sqrt(1 + 4 * jnp.square(conc))
  rho = (r - jnp.sqrt(2. * r)) / (2 * conc)
  s_exact = (1. + jnp.square(rho)) / (2. * rho)
  s_approximate = 1. / jnp.clip(concentration, a_min=1e-7)
  s = jnp.where(use_exact, s_exact, s_approximate)

  def loop_body(arg):
    done, u_in, w, seed, count = arg
    del u_in
    u_seed, v_seed, next_seed = jax.random.split(seed, 3)
    u = jax.random.uniform(
        u_seed, shape=shape, dtype=dtype, minval=-1., maxval=1.
    )
    z = jnp.cos(math.pi * u)
    w = jnp.where(done, w, (1 + s * z) / (s + z))
    y = concentration * (s - w)
    v = jax.random.uniform(v_seed, shape, dtype=dtype, minval=0., maxval=1.)
    # Use `logical_not` to accept all "nan" samples.
    accept = jnp.logical_not(y * jnp.exp(1 - y) < v)
    return jnp.logical_or(accept, done), u, w, next_seed, count + 1

  def loop_cond(arg):
    done, u_in, w, seed, count = arg
    del u_in, w, seed
    # The rejection sampling is actually very efficient. With the worst
    # concentration, about half of the samples are rejected. So only
    # 1 in 1.51e25 samples will ever hit this counter on the worst possible
    # concentration, which is a point way beyond numerical accuracy anyway.
    return jnp.logical_and(jnp.any(jnp.logical_not(done)), count < 100)

  _, u, w, _, _ = jax.lax.while_loop(
      loop_cond,
      loop_body,
      init_val=(
          jnp.zeros(shape, dtype=jnp.bool_),
          jnp.zeros(shape, dtype=dtype),
          jnp.zeros(shape, dtype=dtype),
          seed,
          0
      )
  )
  return jnp.sign(u) * jnp.arccos(jnp.clip(w, a_min=-1., a_max=1.))


# Since rejection sampling does not permit autodiff, add an analytic gradient.
@_von_mises_sample.defjvp
def _von_mises_sample_jvp(
    shape: Union[IntLike, Sequence[IntLike]],
    seed: PRNGKey,
    dtype: jnp.dtype,
    primals: Tuple[Array],
    tangents: Tuple[Array],
) -> Tuple[Array, Array]:
  """Returns the jvp of the von Mises sample operation."""
  concentration, = primals
  dconcentration, = tangents

  concentration = jnp.clip(concentration, a_min=1e-7)
  samples = _von_mises_sample(shape, concentration, seed, dtype)
  vectorized_grad_cdf = jnp.vectorize(
      jax.grad(_von_mises_cdf, argnums=1),
      signature='(),()->()',
      excluded=(2,),
  )
  dcdf_dconcentration = vectorized_grad_cdf(samples, concentration, dtype)

  inv_prob = jnp.exp(-concentration * (jnp.cos(samples) - 1.)) * (
      (2. * math.pi) * jax.scipy.special.i0e(concentration)
  )
  dsamples = dconcentration * (-dcdf_dconcentration * inv_prob)
  return samples, dsamples


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _von_mises_cdf(
    value: Array,
    concentration: Array,
    dtype: jnp.dtype,
) -> Array:
  """Returns the cumulative density function (CDF) of von Mises distribution.

  Denote the density of VonMises(loc=0, concentration=concentration) by p(t).
  The CDF at the point x is defined as int_{-pi}^x p(t) dt when x is in the
  interval [-pi, pi]. Below -pi, the CDF is zero, above pi, it is one.

  The CDF is not available in closed form. Instead, we use the method [1]
  which uses either a series expansion or a Normal approximation, depending on
  the value of concentration.

  We also compute the derivative of the CDF w.r.t. both x and concentration
  using the method described in [2].

  Args:
    value: The point at which to evaluate the CDF.
    concentration: The concentration parameter of the von Mises distribution.
    dtype: Type of the return value.

  Returns:
    The value of the CDF computed elementwise.

  References:
    [1] G. Hill "Algorithm 518: Incomplete Bessel Function I_0. The Von Mises
    Distribution." ACM Transactions on Mathematical Software, 1977.
    [2] Figurnov, M., Mohamed, S. and Mnih, A., "Implicit reparameterization
    gradients." Advances in Neural Information Processing Systems, 31, 2018.
  """
  primals = (value, concentration)
  tangents = (jnp.zeros_like(value), jnp.zeros_like(concentration))
  primal_out, _ = _von_mises_cdf_jvp(dtype, primals, tangents)
  return primal_out


# Use a custom jvp to increase numerical accuracy.
@_von_mises_cdf.defjvp
def _von_mises_cdf_jvp(
    dtype: jnp.dtype,
    primals: Tuple[Array, Array],
    tangents: Tuple[Array, Array],
):
  """Returns the jvp CDF of a von Mises."""
  x, concentration = primals
  dx, dconcentration = tangents
  num_periods = jnp.round(x / (2 * math.pi))
  x = x - (2 * math.pi) * num_periods

  # This is the cutoff-concentration for choosing between the two numerical
  # recipes for computing the CDF. For concentrations larger than
  # `concentration_cutoff`, a Normal approximation is used.
  concentration_cutoff = 10.5

  cdf_series, dcdf_dconcentration_series = _von_mises_cdf_series(
      x, concentration, dtype
  )
  cdf_normal, dcdf_dconcentration_normal = _von_mises_cdf_normal(
      x, concentration, dtype
  )
  use_series = concentration < concentration_cutoff
  cdf = jnp.where(use_series, cdf_series, cdf_normal)
  cdf = cdf + num_periods
  dcdf_dconcentration = jnp.where(
      use_series,
      dcdf_dconcentration_series,
      dcdf_dconcentration_normal,
  )

  prob = jnp.exp(concentration * (jnp.cos(x) - 1.)) / (
      (2. * math.pi) * jax.scipy.special.i0e(concentration)
  )

  return cdf, dconcentration * dcdf_dconcentration + dx * prob


def _von_mises_cdf_series(
    value: Array,
    concentration: Array,
    dtype: jnp.dtype,
    num_terms: int = 20,
) -> Tuple[Array, Array]:
  """Computes the CDF based on a series of `num_terms` terms."""
  rn = jnp.zeros_like(value, dtype=dtype)
  drn_dconcentration = jnp.zeros_like(value, dtype=dtype)
  vn = jnp.zeros_like(value, dtype=dtype)
  dvn_dconcentration = jnp.zeros_like(value, dtype=dtype)

  for n in range(num_terms, 0, -1):
    denominator = 2. * n / concentration + rn
    ddenominator_dk = -2. * n / jnp.square(concentration) + drn_dconcentration
    rn = 1. / denominator
    drn_dconcentration = -ddenominator_dk / jnp.square(denominator)
    multiplier = jnp.sin(n * value) / n + vn
    vn = rn * multiplier
    dvn_dconcentration = (
        drn_dconcentration * multiplier + rn * dvn_dconcentration
    )
  cdf = .5 + value / (2. * math.pi) + vn / math.pi
  dcdf_dconcentration = dvn_dconcentration / math.pi
  # Clip the result to [0, 1].
  cdf_clipped = jnp.clip(cdf, 0., 1.)
  # The clipped values do not depend on concentration anymore, so set their
  # derivative to zero.
  dcdf_dconcentration = (
      dcdf_dconcentration * jnp.logical_and(cdf >= 0., cdf <= 1.)
  )
  return cdf_clipped, dcdf_dconcentration


def _von_mises_cdf_normal(
    value: Array,
    concentration: Array,
    dtype: jnp.dtype,
) -> Tuple[Array, Array]:
  """Computes the CDF, based on a Normal approximation to the von Mises."""
  def cdf_func(value, concentration):
    """A helper function that is passed to value_and_gradient."""
    # z is an "almost Normally distributed" random variable.
    z = (
        jnp.sqrt(2. / math.pi) /
        jax.scipy.special.i0e(concentration) * jnp.sin(.5 * value)
    )
    # This is a correction described in [1].
    # It reduces the error of the Normal approximation.
    z2 = jnp.square(z)
    z3 = z2 * z
    z4 = jnp.square(z2)
    c = 24. * concentration
    c1 = 56.

    xi = z - z3 / jnp.square(
        (c - 2. * z2 - 16.) / 3. -
        (z4 + (7. / 4.) * z2 + 167. / 2.) / (c - c1 - z2 + 3.)
    )
    distrib = normal.Normal(
        loc=jnp.array(0., dtype),
        scale=jnp.array(1., dtype)
    )
    return distrib.cdf(xi)

  vectorized_cdf_with_grad = jnp.vectorize(
      jax.value_and_grad(cdf_func, argnums=1),
      signature='(),()->(),()',
  )
  return vectorized_cdf_with_grad(value, concentration)


def _kl_divergence_vonmises_vonmises(
    dist1: Union[VonMises, tfd.VonMises],
    dist2: Union[VonMises, tfd.VonMises],
    *unused_args, **unused_kwargs,
) -> Array:
  """Batched KL divergence KL(d1 || d2) between von Mises distributions.

  Args:
    dist1: A VonMises distribution.
    dist2: A VonMises distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  i0e_concentration1 = jax.scipy.special.i0e(dist1.concentration)
  i1e_concentration1 = jax.scipy.special.i1e(dist1.concentration)
  i0e_concentration2 = jax.scipy.special.i0e(dist2.concentration)
  return (
      (dist2.concentration - dist1.concentration) +
      jnp.log(i0e_concentration2 / i0e_concentration1) +
      (i1e_concentration1 / i0e_concentration1) * (
          dist1.concentration
          - dist2.concentration * jnp.cos(dist1.loc - dist2.loc)
      )
  )


# Register the KL functions with TFP.
tfd.RegisterKL(VonMises, VonMises)(_kl_divergence_vonmises_vonmises)
tfd.RegisterKL(VonMises, VonMises.equiv_tfp_cls)(
    _kl_divergence_vonmises_vonmises
)
tfd.RegisterKL(VonMises.equiv_tfp_cls, VonMises)(
    _kl_divergence_vonmises_vonmises
)

