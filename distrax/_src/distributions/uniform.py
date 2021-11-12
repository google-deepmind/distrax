# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Uniform distribution."""

import math
from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Uniform(distribution.Distribution):
  """Uniform distribution with `low` and `high` parameters."""

  equiv_tfp_cls = tfd.Uniform

  def __init__(self, low: Numeric = 0., high: Numeric = 1.):
    """Initializes a Uniform distribution.

    Args:
      low: Lower bound.
      high: Upper bound.
    """
    super().__init__()
    self._low = conversion.as_float_array(low)
    self._high = conversion.as_float_array(high)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._low.shape, self._high.shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of the events."""
    return ()

  @property
  def low(self) -> Array:
    """Lower bound."""
    return jnp.broadcast_to(self._low, self.batch_shape)

  @property
  def high(self) -> Array:
    """Upper bound."""
    return jnp.broadcast_to(self._high, self.batch_shape)

  @property
  def range(self) -> Array:
    return self.high - self.low

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return self._batch_shape

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    new_shape = (n,) + self.batch_shape
    uniform = jax.random.uniform(
        key=key, shape=new_shape, dtype=self.range.dtype, minval=0., maxval=1.)
    low = jnp.expand_dims(self._low, range(uniform.ndim - self._low.ndim))
    range_ = jnp.expand_dims(self.range, range(uniform.ndim - self.range.ndim))
    return low + range_ * uniform

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    samples = self._sample_n(key, n)
    log_prob = -jnp.log(self.range)
    log_prob = jnp.repeat(log_prob[None], n, axis=0)
    return samples, log_prob

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    return jnp.log(self.prob(value))

  def prob(self, value: Array) -> Array:
    """See `Distribution.prob`."""
    return jnp.where(
        jnp.logical_or(value < self.low, value > self.high),
        jnp.zeros_like(value),
        jnp.ones_like(value) / self.range)

  def entropy(self) -> Array:
    """Calculates the entropy."""
    return jnp.log(self.range)

  def mean(self) -> Array:
    """Calculates the mean."""
    return (self.low + self.high) / 2.

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self.range) / 12.

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self.range / math.sqrt(12.)

  def median(self) -> Array:
    """Calculates the median."""
    return self.mean()

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    ones = jnp.ones_like(self.range)
    zeros = jnp.zeros_like(ones)
    result_if_not_big = jnp.where(
        value < self.low, zeros, (value - self.low) / self.range)
    return jnp.where(value > self.high, ones, result_if_not_big)

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def __getitem__(self, index) -> 'Uniform':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Uniform(low=self.low[index], high=self.high[index])


def _kl_divergence_uniform_uniform(
    dist1: Union[Uniform, tfd.Uniform],
    dist2: Union[Uniform, tfd.Uniform],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Obtain the KL divergence `KL(dist1 || dist2)` between two Uniforms.

  Note that the KL divergence is infinite if the support of `dist1` is not a
  subset of the support of `dist2`.

  Args:
    dist1: A Uniform distribution.
    dist2: A Uniform distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  return jnp.where(
      jnp.logical_and(dist2.low <= dist1.low, dist1.high <= dist2.high),
      jnp.log(dist2.high - dist2.low) - jnp.log(dist1.high - dist1.low),
      jnp.inf)


# Register the KL functions with TFP.
tfd.RegisterKL(Uniform, Uniform)(_kl_divergence_uniform_uniform)
tfd.RegisterKL(Uniform, Uniform.equiv_tfp_cls)(_kl_divergence_uniform_uniform)
tfd.RegisterKL(Uniform.equiv_tfp_cls, Uniform)(_kl_divergence_uniform_uniform)
