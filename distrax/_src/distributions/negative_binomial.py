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
"""NegativeBinomial distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from jax.lax import lgamma, broadcast_shapes
from jax.random import poisson, split
from jax.nn import log_sigmoid
import jax.numpy as jnp
from distrax import Gamma
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class NegativeBinomial(distribution.Distribution):
  """Negative binomial distribution with parameters `mu` and `phi`"""

  equiv_tfp_cls = tfd.NegativeBinomial

  def __init__(self, total_count: Numeric, logits: Numeric = None, probs: Numeric = None):
    """Negative Binomial distribution

    Args:
      total_count: Positive floating-point array with same dtype as `probs` or `logits`. \
        This represents the number of negative Bernoulli trials to stop at \
        (the `total_count` of failures). Its components should be equal to integer values.
      logits: Logits for the probability of success for independent Negative Binomial \
        distributions. Only one of `logits` or `probs` should be specified.
      probs: Probabilities of success for independent Negative Binomial distributions. \
        Only one of `logits` or `probs` should be specified.
    """
    if (probs is None) == (logits is None):
      raise ValueError(
          'Construct `NegativeBinomial` with `probs` or `logits` but not both.')
    super().__init__()
    self._total_count = conversion.as_float_array(total_count)
    self._logits = conversion.as_float_array(logits)
    self._probs = conversion.as_float_array(probs)
    self._batch_shape = broadcast_shapes(
      self._total_count.shape, self._logits.shape, self._probs.shape
    )

  @classmethod
  def from_mean_dispersion(
    cls,
    mean: Numeric,
    dispersion: Numeric
  ) -> 'NegativeBinomial':
    """Constructs a NegativeBinomial from its mean and dispersion.

    In this parameterization, the dispersion is defined as the reciprocal of the
    total count of failures, i.e. `dispersion = 1 / total_count`.

    Args:
      mean: The mean of the constructed distribution.
      dispersion: The reciprocal of the total_count of the constructed
        distribution.

    Returns:
      neg_bin: A distribution with the given parameterization.
    """
    total_count = jnp.reciprocal(dispersion)
    probs = mean / (mean + total_count)
    return cls(total_count=total_count, probs=probs)

  @property
  def total_count(self) -> Array:
    """Number of negative trials"""
    return jnp.broadcast_to(self._total_count, self.batch_shape)

  @property
  def logits(self) -> Array:
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    if jnp.isnan(self._logits):
      return jnp.log(self.probs) - jnp.log1p(-self.probs) 
    return jnp.broadcast_to(self._logits, self.batch_shape)

  @property
  def probs(self) -> Array:
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    if jnp.isnan(self._probs):
      return jnp.sigmoid(self.logits)
    return jnp.broadcast_to(self._probs, self.batch_shape)

  def mean(self) -> Array:
    """Mean of the distribution"""
    return self.total_count * jnp.exp(self.logits)

  @property
  def dispersion(self) -> Array:
    """Overdispersion of the distribution"""
    return jnp.reciprocal(self.total_count)

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return self._batch_shape

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`.

    Implemented as poisson samples with a gamma distributed rate parameter
    """
    out_shape = (n,) + self.batch_shape
    gamma_key, poisson_key = split(key)
    poisson_rate = Gamma(concentration=self.total_count, rate=jnp.exp(-self.logits))\
        .sample(seed=gamma_key, sample_shape=n)
    gamma_poisson_samples = poisson(lam=poisson_rate, shape=out_shape, key=poisson_key)
    return gamma_poisson_samples


  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    log_unnormalized_prob = (
      (self.total_count * log_sigmoid(-self.logits)) 
      + (log_sigmoid(self.logits) * value)
    )
    log_normalization = (
      lgamma(1. + value)
      + lgamma(self.total_count)
      - lgamma(1. + value + self.total_count)
      + jnp.log(self.total_count + value)
    )
    return log_unnormalized_prob - log_normalization

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()
