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
"""Binomial distribution."""

from typing import Any, Optional, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
from jax import random
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class Binomial(distribution.Distribution):
  """Binomial distribution.

  Binomial distribution with parameters `total_count` (number of trials) and
  `probs` (probability of success per trial).
  """

  equiv_tfp_cls = tfd.Binomial

  def __init__(
      self,
      total_count: Union[int, Array],
      logits: Optional[Numeric] = None,
      probs: Optional[Numeric] = None,
      dtype: Union[jnp.dtype, type[Any]] = int,
  ):
    """Initializes a Binomial distribution.

    Args:
      total_count: Number of trials. Must be a non-negative integer.
      logits: Logit transform of the probability of success, i.e. `probs =
        sigmoid(logits)`. Only one of `logits` or `probs` can be specified.
      probs: Probability of success. Only one of `logits` or `probs` can be
        specified.
      dtype: The type of event samples.
    """
    super().__init__()
    # Validate arguments.
    if (logits is None) == (probs is None):
      raise ValueError(
          'One and exactly one of `logits` and `probs` should be `None`, '
          f'but `logits` is {logits} and `probs` is {probs}.'
      )
    if not (
        jnp.issubdtype(dtype, jnp.integer)
        or jnp.issubdtype(dtype, jnp.floating)
    ):
      raise ValueError(
          f'The dtype of `{self.name}` must be integer or '
          f'floating-point, instead got `{dtype}`.'
      )

    self._total_count = jnp.asarray(total_count, dtype=jnp.int32)
    self._probs = None if probs is None else conversion.as_float_array(probs)
    self._logits = None if logits is None else conversion.as_float_array(logits)
    self._dtype = dtype

    # Check constraints
    if jnp.any(self._total_count < 0):
      raise ValueError('`total_count` must be non-negative.')

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    if self._logits is not None:
      shape = self._logits.shape
    else:
      shape = self._probs.shape
    return jnp.broadcast_shapes(shape, self._total_count.shape)

  @property
  def total_count(self) -> Array:
    """The number of trials."""
    return self._total_count

  @property
  def logits(self) -> Array:
    """The logits of a success."""
    if self._logits is not None:
      return self._logits
    return jnp.log(self._probs) - jnp.log(1 - self._probs)

  @property
  def probs(self) -> Array:
    """The probabilities of a success."""
    if self._probs is not None:
      return self._probs
    return jax.nn.sigmoid(self._logits)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    probs = self.probs
    total_count = self.total_count

    shape = (n,) + self.batch_shape
    samples = random.binomial(key=key, p=probs, n=total_count, shape=shape)
    return samples.astype(self._dtype)

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    value_float = jnp.asarray(value, dtype=jnp.float32)
    total_count_float = self._total_count.astype(jnp.float32)

    # Handle values outside support [0, n] or non-integer values
    is_outside_support = (value_float < 0) | (value_float > total_count_float)
    is_non_integer = value_float != jnp.floor(value_float)
    invalid_values = is_outside_support | is_non_integer

    # Calculations
    # log(p) = -softplus(-logits)
    # log(1-p) = -softplus(logits)
    log_p = -jax.nn.softplus(-self.logits)
    log_1_minus_p = -jax.nn.softplus(self.logits)

    log_unnormalized = math.multiply_no_nan(
        value_float, log_p
    ) + math.multiply_no_nan(total_count_float - value_float, log_1_minus_p)

    # Clip values *only* for lgamma to prevent NaNs.
    # The final `jnp.where` will overwrite these dummy computations.
    safe_value_float = jnp.clip(value_float, 0, total_count_float)

    log_normalization = (
        jax.scipy.special.gammaln(total_count_float + 1.0)
        - jax.scipy.special.gammaln(safe_value_float + 1.0)
        - jax.scipy.special.gammaln(total_count_float - safe_value_float + 1.0)
    )

    log_prob = log_unnormalized - log_normalization

    return jnp.where(invalid_values, -jnp.inf, log_prob)

  def prob(self, value: EventT) -> Array:
    """See `Distribution.prob`."""
    return jnp.exp(self.log_prob(value))

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    # CDF is cumulative sum of PMF from 0 to value.
    # This is implemented using `betainc` (regularized incomplete beta function)
    # which relates Binomial CDF and Beta CDF.
    # CDF(k, n, p) = I_{1-p}(n-k, k+1)
    k = jnp.floor(value)
    n_float = self._total_count.astype(jnp.float32)
    p = self.probs

    # Arguments for betainc
    a = n_float - k
    b = k + 1.0
    x = 1.0 - p

    # Ensure args are positive for betainc
    safe_a = jnp.where(a <= 0, 1.0, a)
    safe_b = jnp.where(b <= 0, 1.0, b)
    cdf_val = jax.scipy.special.betainc(safe_a, safe_b, x)

    # Correct values for out-of-domain k
    cdf_val = jnp.where(k < 0, 0.0, cdf_val)
    cdf_val = jnp.where(k >= n_float, 1.0, cdf_val)

    return cdf_val

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def entropy(self) -> Array:
    """See `Distribution.entropy`."""
    # Entropy is not easily computed in closed form.
    # We can approximate it using the normal distribution for large n.
    # For now, following TFP, we do not implement it.
    raise NotImplementedError('Entropy is not implemented for Binomial.')

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    return self._total_count * self.probs

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    return self._total_count * self.probs * (1 - self.probs)

  def mode(self) -> Array:
    """See `Distribution.mode`."""
    return jnp.floor((self._total_count + 1) * self.probs).astype(self._dtype)

  def __getitem__(self, index) -> 'Binomial':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)

    total_count = self.total_count[index]

    if self._logits is not None:
      return Binomial(
          total_count=total_count, logits=self.logits[index], dtype=self._dtype
      )
    return Binomial(
        total_count=total_count, probs=self.probs[index], dtype=self._dtype
    )
