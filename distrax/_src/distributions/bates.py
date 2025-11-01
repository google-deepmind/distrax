# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Bates distribution."""

from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
# from distrax._src.utils import math as distrax_math # Not using safe_log
import jax
from jax.errors import ConcretizationTypeError
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


def _bates_prob_01(total_count: float, value: Array) -> Array:
  """Helper for prob in [0, 1] range. total_count is static."""
  n = total_count
  x = value

  # Symmetrize around 0.5 for numerical stability
  x_adj = jnp.where(x < 0.5, x, 1.0 - x)
  x_adj = jnp.clip(x_adj, 0.0, 1.0)
  nx = n * x_adj

  is_n_1 = n == 1.0

  # j = floor(nx)
  j = jnp.floor(nx)

  # k = [0, 1, ..., max_j]
  # We need to loop from k=0 to j for each element.
  # Since n is static, max_j is floor(n).
  max_k = int(n)
  k_s = jnp.arange(max_k + 1).astype(value.dtype)  # [0, 1, ..., n]

  # Expand k_s to match batch shape of nx
  # k_s shape (k,), nx shape (...)
  # We want (..., k)
  k_s = jnp.broadcast_to(k_s, nx.shape + (max_k + 1,))
  nx_s = jnp.expand_dims(nx, axis=-1)  # (..., 1)
  n_s = n  # scalar

  # log(nCk) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
  log_comb = (
      jax.lax.lgamma(n_s + 1.0)
      - jax.lax.lgamma(k_s + 1.0)
      - jax.lax.lgamma(n_s - k_s + 1.0)
  )

  # (nx - k)^(n - 1)
  # Ensure base is non-negative for fractional powers if n < 2
  base = jnp.maximum(0.0, nx_s - k_s)
  term_pow = base ** (n_s - 1.0)

  # Sum term: (-1)^k * (nCk) * (nx - k)^(n-1)
  terms = (-1.0) ** k_s * jnp.exp(log_comb) * term_pow

  # We only want to sum terms where k <= j
  # Broadcast j to (..., k)
  j_s = jnp.expand_dims(j, axis=-1)
  valid_terms = jnp.where(k_s <= j_s, terms, 0.0)

  sum_b = jnp.sum(valid_terms, axis=-1)

  # Normalize: sum * n / (n-1)! = sum * n / exp(lgamma(n))
  pdf_01 = sum_b * n / jnp.exp(jax.lax.lgamma(n))

  # Handle n=1 (Uniform) case explicitly.
  pdf_01 = jnp.where(is_n_1, 1.0, pdf_01)

  return pdf_01


def _bates_cdf_01(total_count: float, value: Array) -> Array:
  """Helper for cdf in [0, 1] range. total_count is static."""
  n = total_count
  x = value

  # Symmetrize around 0.5 for numerical stability
  x_adj = jnp.where(x < 0.5, x, 1.0 - x)
  x_adj = jnp.clip(x_adj, 0.0, 1.0)  # Ensure x_adj is in [0, 1]

  nx = n * x_adj

  # j = floor(nx)
  j = jnp.floor(nx)

  is_n_1 = n == 1.0

  # k = [0, 1, ..., max_j]
  max_k = int(n)
  k_s = jnp.arange(max_k + 1).astype(value.dtype)  # [0, 1, ..., n]

  # Expand k_s to match batch shape of nx
  k_s = jnp.broadcast_to(k_s, nx.shape + (max_k + 1,))
  nx_s = jnp.expand_dims(nx, axis=-1)  # (..., 1)
  n_s = n  # scalar

  # Calculate (n choose k)
  log_comb = (
      jax.lax.lgamma(n_s + 1.0)
      - jax.lax.lgamma(k_s + 1.0)
      - jax.lax.lgamma(n_s - k_s + 1.0)
  )

  # (nx - k)^n
  base = jnp.maximum(0.0, nx_s - k_s)
  term_pow = base**n_s

  # Sum term: (-1)^k * (nCk) * (nx - k)^n
  terms = (-1.0) ** k_s * jnp.exp(log_comb) * term_pow

  # We only want to sum terms where k <= j
  j_s = jnp.expand_dims(j, axis=-1)
  valid_terms = jnp.where(k_s <= j_s, terms, 0.0)

  # Sum over k for each segment
  sum_b = jnp.sum(valid_terms, axis=-1)

  # Normalize: sum / n! = sum / exp(lgamma(n + 1))
  cdf_adj = sum_b / jnp.exp(jax.lax.lgamma(n + 1.0))

  # Handle n=1 (Uniform) case
  cdf_adj = jnp.where(is_n_1, x_adj, cdf_adj)

  # Adjust for symmetry: cdf(x) = 1 - cdf(1 - x) for x > 0.5
  cdf_01 = jnp.where(x > 0.5, 1.0 - cdf_adj, cdf_adj)

  return cdf_01


class Bates(distribution.Distribution):
  """Bates distribution.

  The Bates distribution is the distribution of the average of `total_count`
  independent samples from `Uniform(low, high)`.
  """

  equiv_tfp_cls = tfd.Bates

  def __init__(
      self,
      total_count: Union[int, float, Array],
      low: Numeric = 0.0,
      high: Numeric = 1.0,
  ):
    """Initializes a Bates distribution.

    Args:
      total_count: Non-negative integer-valued number of samples. **NOTE**:
        Unlike TFP, `total_count` must be a static Python int/float or a scalar
        JAX array with a static value. Batched `total_count` is not supported
        due to JAX's static shape requirements for sampling.
      low: Lower bound of the uniform samples.
      high: Upper bound of the uniform samples.
    """
    super().__init__()

    # Try to statically get the int value
    try:
      # This will work for Python int/float
      self._total_count_static = int(total_count)
    except (TypeError, ConcretizationTypeError):
      # This might be a JAX array.
      # This will raise ConcretizationError if it's not static.
      try:
        self._total_count_static = int(jnp.asarray(total_count))
      except (TypeError, ConcretizationTypeError) as exc:
        raise ValueError(
            "`total_count` for distrax.Bates must be a static Python"
            " int/float or a statically-valued scalar JAX Array (e.g., from"
            " `jnp.array(5.0)`). Batched/dynamic `total_count` is not"
            " supported due to JAX's static shape requirements for sampling."
        ) from exc

    if self._total_count_static <= 0:
      raise ValueError("`total_count` must be positive.")

    self._total_count_float = float(self._total_count_static)

    # self._total_count is for API compatibility (prob, cdf, etc)
    # It's a scalar array.
    self._total_count = conversion.as_float_array(self._total_count_float)
    self._low = conversion.as_float_array(low)
    self._high = conversion.as_float_array(high)

    # batch_shape ignores total_count's shape, as it must be scalar
    self._batch_shape = jax.lax.broadcast_shapes(
        self._low.shape, self._high.shape
    )

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of the events."""
    return ()

  @property
  def total_count(self) -> Array:
    """Number of `Uniform` trials used to construct a sample."""
    # This returns a scalar array, which will broadcast with batch_shape.
    return self._total_count

  @property
  def low(self) -> Array:
    """Lower bound of the support."""
    return jnp.broadcast_to(self._low, self.batch_shape)

  @property
  def high(self) -> Array:
    """Upper bound of the support."""
    return jnp.broadcast_to(self._high, self.batch_shape)

  @property
  def range(self) -> Array:
    return self.high - self.low

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return self._batch_shape

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""

    # Sample (n, *batch_shape, total_count) uniforms
    sample_shape = (n,) + self.batch_shape + (self._total_count_static,)

    uniform_samples = jax.random.uniform(
        key, shape=sample_shape, dtype=self.low.dtype
    )

    # Take the mean over the last axis
    means = jnp.mean(uniform_samples, axis=-1)  # Shape (n, *batch_shape)

    # Shift/scale from (0, 1) to (low, high).
    # low/range have shape (*batch_shape), need to be expanded for 'n'
    low = jnp.broadcast_to(self.low, (n,) + self.batch_shape)
    range_ = jnp.broadcast_to(self.range, (n,) + self.batch_shape)

    return low + range_ * means

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    return jnp.log(self.prob(value))

  def prob(self, value: EventT) -> Array:
    """See `Distribution.prob`."""
    value_scaled = (value - self.low) / self.range

    # total_count is passed as a static float
    pdf_01 = _bates_prob_01(self._total_count_float, value_scaled)
    prob = pdf_01 / self.range

    # Set prob to 0 outside [low, high]
    return jnp.where(
        jnp.logical_or(value < self.low, value > self.high), 0.0, prob
    )

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    value_scaled = (value - self.low) / self.range

    # total_count is passed as a static float
    cdf_01 = _bates_cdf_01(self._total_count_float, value_scaled)

    # Clamp to [0, 1]
    return jnp.clip(cdf_01, 0.0, 1.0)

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def mean(self) -> Array:
    """Calculates the mean."""
    return (self.low + self.high) / 2.0

  def variance(self) -> Array:
    """Calculates the variance."""
    # self.total_count is a scalar array, broadcasts with self.range
    return jnp.square(self.range) / (12.0 * self.total_count)

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return jnp.sqrt(self.variance())

  def mode(self) -> Array:
    """Calculates the mode."""
    # For n > 1, the mode is the mean.
    # For n = 1 (Uniform), any value in (low, high) is a mode.
    # We follow TFP and return the mean.
    return self.mean()

  def median(self) -> Array:
    """Calculates the median."""
    # For a symmetric distribution, median is the mean.
    return self.mean()

  def __getitem__(self, index) -> "Bates":
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    # total_count is static and scalar, so it is not indexed.
    return Bates(
        total_count=self._total_count_static,
        low=self.low[index],
        high=self.high[index],
    )
