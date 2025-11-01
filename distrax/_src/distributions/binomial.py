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
  `probs` (probability of success for each trial).
  """

  equiv_tfp_cls = tfd.Binomial

  def __init__(
      self,
      total_count: Numeric,
      logits: Optional[Numeric] = None,
      probs: Optional[Numeric] = None,
      dtype: Union[jnp.dtype, type[Any]] = int,
  ):
    """Initializes a Binomial distribution.

    Args:
      total_count: Number of trials. Must be a non-negative integer.
      logits: Log-odds of success.
      probs: Probability of success.
      dtype: The type of the event samples.
    """
    super().__init__()
    if logits is None and probs is None:
      raise ValueError("Either `logits` or `probs` must be specified.")
    if logits is not None and probs is not None:
      raise ValueError("Only one of `logits` or `probs` can be specified.")

    self._total_count = jnp.asarray(total_count)
    if not jnp.issubdtype(self._total_count.dtype, jnp.integer):
      raise ValueError(
          f"Total count must be of integer type, got {self._total_count.dtype}."
      )

    if logits is not None:
      self._logits = conversion.as_float_array(logits)
      if not isinstance(self._logits, jax.Array):
        self._logits = jnp.asarray(self._logits)  # Ensure JAX array
      self._probs = jax.nn.sigmoid(self._logits)
    else:
      self._probs = conversion.as_float_array(probs)
      if not isinstance(self._probs, jax.Array):
        self._probs = jnp.asarray(self._probs)  # Ensure JAX array
      # Logit implementation from bernoulli.py
      self._logits = jnp.log(self._probs) - jnp.log1p(-self._probs)

    self._dtype = dtype

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def total_count(self) -> Array:
    """Number of trials."""
    return self._total_count

  @property
  def logits(self) -> Array:
    """Log-odds of success."""
    return self._logits

  @property
  def probs(self) -> Array:
    """Probability of success."""
    return self._probs

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key, subkey = random.split(key)

    # Use tfp.distributions.Binomial's sampler
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )

    samples = tfp_dist.sample(sample_shape=(n,), seed=subkey)
    return samples.astype(self._dtype)

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    samples = self._sample_n(key, n)
    log_prob = self.log_prob(samples)
    return samples, log_prob

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )
    return tfp_dist.log_prob(value)

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )
    return tfp_dist.cdf(value)

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )
    return tfp_dist.log_cdf(value)

  def survival_function(self, value: EventT) -> Array:
    """See `Distribution.survival_function`."""
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )
    return tfp_dist.survival_function(value)

  def log_survival_function(self, value: EventT) -> Array:
    """See `Distribution.log_survival_function`."""
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )
    return tfp_dist.log_survival_function(value)

  def entropy(self) -> Array:
    """See `Distribution.entropy`."""
    tfp_dist = self.equiv_tfp_cls(
        total_count=self._total_count.astype(self._logits.dtype),
        logits=self._logits,
    )
    return tfp_dist.entropy()

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    return self._total_count * self._probs

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    return self._total_count * self._probs * (1.0 - self._probs)

  def stddev(self) -> Array:
    """See `Distribution.stddev`."""
    return jnp.sqrt(self.variance())

  def mode(self) -> Array:
    """See `Distribution.mode`."""
    return jnp.floor(self._total_count * self._probs)

  def __getitem__(self, index: Any) -> "Binomial":
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Binomial(
        total_count=self._total_count[index],
        logits=self._logits[index],
        dtype=self._dtype,
    )
