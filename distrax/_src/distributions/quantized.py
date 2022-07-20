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

from typing import cast, Optional, Tuple

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


class Quantized(
    base_distribution.Distribution[Array, Tuple[int, ...], jnp.dtype],):
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
      low: Lowest possible quantized value, such that samples are `y >=
        ceil(low)`. Its shape must broadcast with the shape of samples from
        `distribution` and must not result in additional batch dimensions after
        broadcasting.
      high: Highest possible quantized value, such that samples are `y <=
        floor(high)`. Its shape must broadcast with the shape of samples from
        `distribution` and must not result in additional batch dimensions after
        broadcasting.
    """
    self._dist: base_distribution.Distribution[Array, Tuple[
        int, ...], jnp.dtype] = conversion.as_distribution(distribution)
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
  def distribution(
      self
  ) -> base_distribution.Distribution[Array, Tuple[int, ...], jnp.dtype]:
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
    event_shape = self.distribution.event_shape
    # TODO(b/149413467): Remove explicit casting when resolved.
    return cast(Tuple[int, ...], event_shape)

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
    log_sf = self.distribution.log_survival_function(samples)
    log_sf_m1 = self.distribution.log_survival_function(samples - 1.)
    if self.high is not None:
      # `samples - 1.` is definitely lower than `high`.
      log_cdf = jnp.where(samples < self.high, log_cdf, 0.)
      log_sf = jnp.where(samples < self.high, log_sf, -jnp.inf)
    if self.low is not None:
      # `samples` is definitely greater than or equal to `low`.
      log_cdf_m1 = jnp.where(samples - 1. < self.low, -jnp.inf, log_cdf_m1)
      log_sf_m1 = jnp.where(samples - 1. < self.low, 0., log_sf_m1)
    # Use the survival function instead of the CDF when its value is smaller,
    # which happens to the right of the median of the distribution.
    big = jnp.where(log_sf < log_cdf, log_sf_m1, log_cdf)
    small = jnp.where(log_sf < log_cdf, log_sf, log_cdf_m1)
    log_probs = math.log_expbig_minus_expsmall(big, small)
    return samples, log_probs

  def log_prob(self, value: Array) -> Array:
    """Calculates the log probability of an event.

    This implementation differs slightly from the one in TFP, as it returns
    `-jnp.inf` on non-integer values instead of returning the log prob of the
    floor of the input. In addition, this implementation also returns `-jnp.inf`
    on inputs that are outside the support of the distribution (as opposed to
    `nan`, like TFP does). On other integer values, both implementations are
    identical.

    Similar to TFP, the log prob is computed using either the CDF or the
    survival function to improve numerical stability. With infinite precision
    the two computations would be equal.

    Args:
      value: An event.

    Returns:
      The log probability log P(value).
    """
    log_cdf = self.log_cdf(value)
    log_cdf_m1 = self.log_cdf(value - 1.)
    log_sf = self.log_survival_function(value)
    log_sf_m1 = self.log_survival_function(value - 1.)
    # Use the survival function instead of the CDF when its value is smaller,
    # which happens to the right of the median of the distribution.
    big = jnp.where(log_sf < log_cdf, log_sf_m1, log_cdf)
    small = jnp.where(log_sf < log_cdf, log_sf, log_cdf_m1)
    log_probs = math.log_expbig_minus_expsmall(big, small)

    # Return -inf when evaluating on non-integer value.
    is_integer = jnp.where(value > jnp.floor(value), False, True)
    log_probs = jnp.where(is_integer, log_probs, -jnp.inf)

    # Return -inf and not NaN when outside of [low, high].
    # If the CDF is used, `value > high` is already treated correctly;
    # to fix the return value for `value < low` we test whether `log_cdf` is
    # finite; `log_sf_m1` will be `0.` in this regime.
    # If the survival function is used the reverse case applies; to fix the
    # case `value > high` we test whether `log_sf_m1` is finite; `log_cdf` will
    # be `0.` in this regime.
    is_outside = jnp.logical_or(jnp.isinf(log_cdf), jnp.isinf(log_sf_m1))
    log_probs = jnp.where(is_outside, -jnp.inf, log_probs)

    return log_probs

  def prob(self, value: Array) -> Array:
    """Calculates the probability of an event.

    This implementation differs slightly from the one in TFP, as it returns 0
    on non-integer values instead of returning the prob of the floor of the
    input. It is identical for integer values.

    Similar to TFP, the probability is computed using either the CDF or the
    survival function to improve numerical stability. With infinite precision
    the two computations would be equal.

    Args:
      value: An event.

    Returns:
      The probability P(value).
    """
    cdf = self.cdf(value)
    cdf_m1 = self.cdf(value - 1.)
    sf = self.survival_function(value)
    sf_m1 = self.survival_function(value - 1.)
    # Use the survival function instead of the CDF when its value is smaller,
    # which happens to the right of the median of the distribution.
    probs = jnp.where(sf < cdf, sf_m1 - sf, cdf - cdf_m1)

    # Return 0. when evaluating on non-integer value.
    is_integer = jnp.where(value > jnp.floor(value), False, True)
    probs = jnp.where(is_integer, probs, 0.)
    return probs

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    # The log CDF of a quantized distribution is piecewise constant on half-open
    # intervals:
    #    ... [n-2   n-1) [n-1   n) [n   n+1) [n+1   n+2) ...
    # with log CDF(n) <= log CDF(n+1), because the distribution only has mass on
    # integer values. Therefore: P[Y <= value] = P[Y <= floor(value)].
    y = jnp.floor(value)
    result = self.distribution.log_cdf(y)
    # Update result outside of the interval [low, high].
    if self.low is not None:
      result = jnp.where(y < self.low, -jnp.inf, result)
    if self.high is not None:
      result = jnp.where(y < self.high, result, 0.)
    return result

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    # The CDF of a quantized distribution is piecewise constant on half-open
    # intervals:
    #    ... [n-2   n-1) [n-1   n) [n   n+1) [n+1   n+2) ...
    # with CDF(n) <= CDF(n+1), because the distribution only has mass on integer
    # values. Therefore: P[Y <= value] = P[Y <= floor(value)].
    y = jnp.floor(value)
    result = self.distribution.cdf(y)
    # Update result outside of the interval [low, high].
    if self.low is not None:
      result = jnp.where(y < self.low, 0., result)
    if self.high is not None:
      result = jnp.where(y < self.high, result, 1.)
    return result

  def log_survival_function(self, value: Array) -> Array:
    """Calculates the log of the survival function of an event.

    This implementation differs slightly from TFP, in that it returns the
    correct log of the survival function for non-integer values, that is, it
    always equates to `log(1 - CDF(value))`. It is identical for integer values.

    Args:
      value: An event.

    Returns:
      The log of the survival function `log P[Y > value]`.
    """
    # The log of the survival function of a quantized distribution is piecewise
    # constant on half-open intervals:
    #    ... [n-2   n-1) [n-1   n) [n   n+1) [n+1   n+2) ...
    # with log sf(n) >= log sf(n+1), because the distribution only has mass on
    # integer values. Therefore: log P[Y > value] = log P[Y > floor(value)].
    y = jnp.floor(value)
    result = self.distribution.log_survival_function(y)
    # Update result outside of the interval [low, high].
    if self._low is not None:
      result = jnp.where(y < self._low, 0., result)
    if self._high is not None:
      result = jnp.where(y < self._high, result, -jnp.inf)
    return result

  def survival_function(self, value: Array) -> Array:
    """Calculates the survival function of an event.

    This implementation differs slightly from TFP, in that it returns the
    correct survival function for non-integer values, that is, it always
    equates to `1 - CDF(value)`. It is identical for integer values.

    Args:
      value: An event.

    Returns:
      The survival function `P[Y > value]`.
    """
    # The survival function of a quantized distribution is piecewise
    # constant on half-open intervals:
    #    ... [n-2   n-1) [n-1   n) [n   n+1) [n+1   n+2) ...
    # with sf(n) >= sf(n+1), because the distribution only has mass on
    # integer values. Therefore: P[Y > value] = P[Y > floor(value)].
    y = jnp.floor(value)
    result = self.distribution.survival_function(y)
    # Update result outside of the interval [low, high].
    if self._low is not None:
      result = jnp.where(y < self._low, 1., result)
    if self._high is not None:
      result = jnp.where(y < self._high, result, 0.)
    return result

  def __getitem__(self, index) -> 'Quantized':
    """See `Distribution.__getitem__`."""
    index = base_distribution.to_batch_shape_index(self.batch_shape, index)
    low = None if self._low is None else self.low[index]
    high = None if self._high is None else self.high[index]
    return Quantized(distribution=self.distribution[index], low=low, high=high)
