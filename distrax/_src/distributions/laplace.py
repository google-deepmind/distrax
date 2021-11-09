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
"""Laplace distribution."""

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


class Laplace(distribution.Distribution):
  """Laplace distribution with location `loc` and `scale` parameters."""

  equiv_tfp_cls = tfd.Laplace

  def __init__(self, loc: Numeric, scale: Numeric):
    """Initializes a Laplace distribution.

    Args:
      loc: Mean of the distribution.
      scale: Spread of the distribution.
    """
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._loc.shape, self._scale.shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Scale of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

  def _sample_from_std_laplace(self, key: PRNGKey, n: int) -> Array:
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._loc, self._scale)
    return jax.random.laplace(key, shape=out_shape, dtype=dtype)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    rnd = self._sample_from_std_laplace(key, n)
    return self._loc + self._scale * rnd

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    rnd = self._sample_from_std_laplace(key, n)
    samples = self._loc + self._scale * rnd
    log_prob = -jnp.abs(rnd) - math.log(2.) - jnp.log(self._scale)
    return samples, log_prob

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    norm_value = self._standardize(value)
    return -jnp.abs(norm_value) - math.log(2.) - jnp.log(self._scale)

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    return math.log(2.) + 1. + jnp.log(self.scale)

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    norm_value = self._standardize(value)
    return 0.5 - 0.5 * jnp.sign(norm_value) * jnp.expm1(-jnp.abs(norm_value))

  def _standardize(self, value: Array) -> Array:
    return (value - self._loc) / self._scale

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    norm_value = self._standardize(value)
    lower_value = norm_value - math.log(2.)
    exp_neg_norm_value = jnp.exp(-jnp.abs(norm_value))
    upper_value = jnp.log1p(-0.5 * exp_neg_norm_value)
    return jnp.where(jnp.less_equal(norm_value, 0.), lower_value, upper_value)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return math.sqrt(2.) * self.scale

  def variance(self) -> Array:
    """Calculates the variance."""
    return 2. * jnp.square(self.scale)

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def median(self) -> Array:
    """Calculates the median."""
    return self.mean()

  def __getitem__(self, index) -> 'Laplace':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Laplace(loc=self.loc[index], scale=self.scale[index])


def _kl_divergence_laplace_laplace(
    dist1: Union[Laplace, tfd.Laplace],
    dist2: Union[Laplace, tfd.Laplace],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Batched KL divergence KL(dist1 || dist2) between two Laplace distributions.

  Args:
    dist1: A Laplace distribution.
    dist2: A Laplace distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  distance = jnp.abs(dist1.loc - dist2.loc)
  diff_log_scale = jnp.log(dist1.scale) - jnp.log(dist2.scale)
  return (- diff_log_scale +
          distance / dist2.scale - 1. +
          jnp.exp(-distance / dist1.scale + diff_log_scale))


# Register the KL functions with TFP.
tfd.RegisterKL(Laplace, Laplace)(_kl_divergence_laplace_laplace)
tfd.RegisterKL(Laplace, Laplace.equiv_tfp_cls)(_kl_divergence_laplace_laplace)
tfd.RegisterKL(Laplace.equiv_tfp_cls, Laplace)(_kl_divergence_laplace_laplace)
