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
"""Quantized distribution."""

from typing import Optional, Tuple

import chex
from distrax._src.distributions import distribution as base_distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
DistributionLike = base_distribution.DistributionLike
DistributionT = base_distribution.DistributionT


class Quantized(base_distribution.Distribution):
  """Distribution representing the quantization `Y = ceil(X)`.

  Given an input distribution `p(x)` over a univariate random variable `X`,
  sampling from a `Quantized` distribution amounts to sampling `x ~ p(x)` and
  then setting `y = ceil(x)`. The returned samples are integer-valued and of the
  same `dtype` as the base distribution.
  """

  equiv_tfp_cls = tfd.QuantizedDistribution

  def __init__(self,
               distribution: DistributionLike,
               low: Optional[Numeric] = None,
               high: Optional[Numeric] = None):
    """Initializes a Quantized distribution.

    Args:
      distribution: The base distribution to be quantized.
      low: Lowest possible quantized value, such that samples are
        `y >= ceil(low)`. Its shape must broadcast with the shape of samples
        from `distribution` and must not result in additional batch dimensions
        after broadcasting.
      high: Highest possible quantized value, such that samples are
        `y <= floor(high)`. Its shape must broadcast with the shape of samples
        from `distribution` and must not result in additional batch dimensions
        after broadcasting.
    """
    self._dist = conversion.as_distribution(distribution)
    if self._dist.event_shape:
      raise ValueError(f'The base distribution must be univariate, but its '
                       f'`event_shape` is {self._dist.event_shape}.')
    dtype = self._dist.dtype
    if low is None:
      self._low = None
    else:
      self._low = jnp.asarray(jnp.ceil(low), dtype=dtype)
      if len(self._low.shape) > len(self._dist.batch_shape):
        raise ValueError('The parameter `low` must not result in additional '
                         'batch dimensions.')
    if high is None:
      self._high = None
    else:
      self._high = jnp.asarray(jnp.floor(high), dtype=dtype)
      if len(self._high.shape) > len(self._dist.batch_shape):
        raise ValueError('The parameter `high` must not result in additional '
                         'batch dimensions.')
    super().__init__()

  @property
  def distribution(self) -> DistributionT:
    """Base distribution `p(x)`."""
    return self._dist

  @property
  def low(self) -> Optional[Array]:
    """Lowest value that quantization returns."""
    if self._low is None:
      return None
    return jnp.broadcast_to(self._low, self.batch_shape + self.event_shape)

  @property
  def high(self) -> Optional[Array]:
    """Highest value that quantization returns."""
    if self._high is None:
      return None
    return jnp.broadcast_to(self._high, self.batch_shape + self.event_shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return self.distribution.event_shape

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self.distribution.batch_shape

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    samples = self.distribution.sample(seed=key, sample_shape=n)
    samples = jnp.ceil(samples)

    # Apply overflow and underflow conditions.
    if self.low is not None:
      samples = jnp.where(samples < self.low, self.low, samples)
    if self.high is not None:
      samples = jnp.where(samples > self.high, self.high, samples)

    return samples

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    samples = self._sample_n(key, n)
    log_cdf = self.distribution.log_cdf(samples)
    log_cdf_m1 = self.distribution.log_cdf(samples - 1.)
    if self.high is not None:
      log_cdf = jnp.where(samples < self.high, log_cdf, 0.)
    if self.low is not None:
      log_cdf_m1 = jnp.where(samples - 1. < self.low, -jnp.inf, log_cdf_m1)
    log_probs = math.log_expbig_minus_expsmall(log_cdf, log_cdf_m1)
    return samples, log_probs

  def log_prob(self, value: Array) -> Array:
    """Calculates the log probability of an event.

    This implementation differs slightly from the one in TFP, as it returns
    `-jnp.inf` on non-integer values instead of returning the log prob of the
    floor of the input. In addition, this implementation also returns `-jnp.inf`
    on inputs that are outside the support of the distribution (as opposed to
    `nan`, like TFP does). On other integer values, both implementations are
    identical.

    Args:
      value: An event.

    Returns:
      The log probability log P(value).
    """
    is_integer = jnp.where(value > jnp.floor(value), False, True)
    log_cdf = self.log_cdf(value)
    log_cdf_m1 = self.log_cdf(value - 1.)
    log_probs = math.log_expbig_minus_expsmall(log_cdf, log_cdf_m1)
    return jnp.where(jnp.isinf(log_cdf), -jnp.inf,
                     jnp.where(is_integer, log_probs, -jnp.inf))

  def prob(self, value: Array) -> Array:
    """Calculates the probability of an event.

    This implementation differs slightly from the one in TFP, as it returns 0
    on non-integer values instead of returning the prob of the floor of the
    input. It is identical for integer values.

    Args:
      value: An event.

    Returns:
      The probability P(value).
    """
    is_integer = jnp.where(value > jnp.floor(value), False, True)
    cdf = self.cdf(value)
    cdf_m1 = self.cdf(value - 1.)
    probs = cdf - cdf_m1
    return jnp.where(is_integer, probs, 0.)

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    y = jnp.floor(value)
    result = self.distribution.log_cdf(y)
    if self.low is not None:
      result = jnp.where(y < self.low, -jnp.inf, result)
    if self.high is not None:
      result = jnp.where(y < self.high, result, 0.)
    return result

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    y = jnp.floor(value)
    result = self.distribution.cdf(y)
    if self.low is not None:
      result = jnp.where(y < self.low, 0., result)
    if self.high is not None:
      result = jnp.where(y < self.high, result, 1.)
    return result
