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
"""Clipped distributions."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution as base_distribution
from distrax._src.distributions import logistic
from distrax._src.distributions import normal
from distrax._src.utils import conversion
import jax.numpy as jnp


Array = chex.Array
PRNGKey = chex.PRNGKey
Numeric = chex.Numeric
DistributionLike = base_distribution.DistributionLike


class Clipped(base_distribution.Distribution):
  """A clipped distribution."""

  def __init__(
      self,
      distribution: DistributionLike,
      minimum: Numeric,
      maximum: Numeric):
    """Wraps a distribution clipping samples out of `[minimum, maximum]`.

    The samples outside of `[minimum, maximum]` are clipped to the boundary.
    The log probability of samples outside of this range is `-inf`.

    Args:
      distribution: a Distrax / TFP distribution to be wrapped.
      minimum: can be a `scalar` or `vector`; if a vector, must have fewer dims
        than `distribution.batch_shape` and must be broadcastable to it.
      maximum: can be a `scalar` or `vector`; if a vector, must have fewer dims
        than `distribution.batch_shape` and must be broadcastable to it.
    """
    super().__init__()
    if distribution.event_shape:
      raise ValueError('The wrapped distribution must have event shape ().')
    if (jnp.array(minimum).ndim > len(distribution.batch_shape) or
        jnp.array(maximum).ndim > len(distribution.batch_shape)):
      raise ValueError(
          'The minimum and maximum clipping boundaries must be scalars or'
          'vectors with fewer dimensions as the batch_shape of distribution:'
          'i.e. we can broadcast min/max to batch_shape but not viceversa.')
    self._distribution = conversion.as_distribution(distribution)
    self._minimum = jnp.broadcast_to(minimum, self._distribution.batch_shape)
    self._maximum = jnp.broadcast_to(maximum, self._distribution.batch_shape)
    self._log_prob_minimum = self._distribution.log_cdf(minimum)
    self._log_prob_maximum = self._distribution.log_survival_function(maximum)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    raw_sample = self._distribution.sample(seed=key, sample_shape=[n])
    return jnp.clip(raw_sample, self._minimum, self._maximum)

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    samples = self._sample_n(key, n)
    return samples, self.log_prob(samples)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    # The log_prob can be used to compute expectations by explicitly integrating
    # over the discrete and continuous elements.
    # Info about mixed distributions:
    # http://www.randomservices.org/random/dist/Mixed.html
    log_prob = jnp.where(
        jnp.equal(value, self._minimum),
        self._log_prob_minimum,
        jnp.where(jnp.equal(value, self._maximum),
                  self._log_prob_maximum,
                  self._distribution.log_prob(value)))
    # Giving -inf log_prob outside the boundaries.
    return jnp.where(
        jnp.logical_or(value < self._minimum, value > self._maximum),
        -jnp.inf,
        log_prob)

  @property
  def minimum(self) -> Numeric:
    return self._minimum

  @property
  def maximum(self) -> Numeric:
    return self._maximum

  @property
  def distribution(self) -> DistributionLike:
    return self._distribution

  @property
  def event_shape(self) -> Tuple[int, ...]:
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return self._distribution.batch_shape

  def __getitem__(self, index) -> 'Clipped':
    """See `Distribution.__getitem__`."""
    index = base_distribution.to_batch_shape_index(self.batch_shape, index)
    return Clipped(
        distribution=self.distribution[index],
        minimum=self.minimum[index],
        maximum=self.maximum[index])


class ClippedNormal(Clipped):
  """A clipped normal distribution."""

  def __init__(
      self, loc: Numeric, scale: Numeric, minimum: Numeric, maximum: Numeric):
    distribution = normal.Normal(loc=loc, scale=scale)
    super().__init__(distribution, minimum=minimum, maximum=maximum)


class ClippedLogistic(Clipped):
  """A clipped logistic distribution."""

  def __init__(
      self, loc: Numeric, scale: Numeric, minimum: Numeric, maximum: Numeric):
    distribution = logistic.Logistic(loc=loc, scale=scale)
    super().__init__(distribution, minimum=minimum, maximum=maximum)
