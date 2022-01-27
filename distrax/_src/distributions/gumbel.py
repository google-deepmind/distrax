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
"""Gumbel distribution."""

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


class Gumbel(distribution.Distribution):
  """Gumbel distribution with location `loc` and `scale` parameters."""

  equiv_tfp_cls = tfd.Gumbel

  def __init__(self, loc: Numeric, scale: Numeric):
    """Initializes a Gumbel distribution.

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

  def _standardize(self, value: Array) -> Array:
    """Standardizes the input `value` in location and scale."""
    return (value - self._loc) / self._scale

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    z = self._standardize(value)
    return -(z + jnp.exp(-z)) - jnp.log(self._scale)

  def _sample_from_std_gumbel(self, key: PRNGKey, n: int) -> Array:
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._loc, self._scale)
    return jax.random.gumbel(key, shape=out_shape, dtype=dtype)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    rnd = self._sample_from_std_gumbel(key, n)
    return self._scale * rnd + self._loc

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    rnd = self._sample_from_std_gumbel(key, n)
    samples = self._scale * rnd + self._loc
    log_prob = -(rnd + jnp.exp(-rnd)) - jnp.log(self._scale)
    return samples, log_prob

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    return jnp.log(self._scale) + 1. + jnp.euler_gamma

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    z = self._standardize(value)
    return -jnp.exp(-z)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._loc + self._scale * jnp.euler_gamma

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self._scale * jnp.ones_like(self._loc) * jnp.pi / math.sqrt(6.)

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self._scale * jnp.ones_like(self._loc) * jnp.pi) / 6.

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.loc

  def median(self) -> Array:
    """Calculates the median."""
    return self._loc - self._scale * math.log(math.log(2.))

  def __getitem__(self, index) -> 'Gumbel':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Gumbel(loc=self.loc[index], scale=self.scale[index])


def _kl_divergence_gumbel_gumbel(
    dist1: Union[Gumbel, tfd.Gumbel],
    dist2: Union[Gumbel, tfd.Gumbel],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Batched KL divergence KL(dist1 || dist2) between two Gumbel distributions.

  Args:
    dist1: A Gumbel distribution.
    dist2: A Gumbel distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  return (jnp.log(dist2.scale) - jnp.log(dist1.scale) + jnp.euler_gamma *
          (dist1.scale / dist2.scale - 1.) +
          jnp.expm1((dist2.loc - dist1.loc) / dist2.scale +
                    jax.lax.lgamma(dist1.scale / dist2.scale + 1.)) +
          (dist1.loc - dist2.loc) / dist2.scale)


# Register the KL functions with TFP.
tfd.RegisterKL(Gumbel, Gumbel)(_kl_divergence_gumbel_gumbel)
tfd.RegisterKL(Gumbel, Gumbel.equiv_tfp_cls)(_kl_divergence_gumbel_gumbel)
tfd.RegisterKL(Gumbel.equiv_tfp_cls, Gumbel)(_kl_divergence_gumbel_gumbel)
