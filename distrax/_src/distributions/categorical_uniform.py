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
"""Categorical-uniform distributions."""

from typing import Tuple

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
from distrax._src.distributions import mixture_same_family
from distrax._src.distributions import uniform
import jax
import jax.numpy as jnp


Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class CategoricalUniform(distribution.Distribution):
  """Mixture Categorical-Uniform distribution with reparameterization trick."""

  def __init__(
      self,
      *,
      high: Numeric,
      low: Numeric,
      logits: Array,
  ) -> None:
    """Initializer."""
    super().__init__()
    self._low = low
    self._high = high
    self._logits = logits

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return self._get_mixture().event_shape

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._logits.shape[:-1]

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    return jax.vmap(self._sample)(jax.random.split(key, n))

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    return self._get_mixture().log_prob(value)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._get_mixture().mean()

  def variance(self) -> Array:
    """Calculates the variance."""
    return self._get_mixture().variance()

  def __getitem__(self, key) -> 'CategoricalUniform':
    """See `Distribution.__getitem__`."""
    return CategoricalUniform(
        high=self.high[key], low=self.low[key], logits=self.logits[key])

  def _sample(self, key: PRNGKey) -> Array:
    """Draws one sample."""
    quantile = jax.random.uniform(key, self.batch_shape)
    return self._inverse_cdf(quantile)

  def _get_category_limits(self) -> Array:
    """Gets limits for each category."""
    return jnp.linspace(self.low, self.high, self.num_bins + 1, axis=-1)

  def _get_mixture(self) -> mixture_same_family.MixtureSameFamily:
    """Gets a mixture distribution."""
    limits = self._get_category_limits()
    return mixture_same_family.MixtureSameFamily(
        components_distribution=uniform.Uniform(
            low=limits[..., :-1], high=limits[..., 1:]),
        mixture_distribution=categorical.Categorical(logits=self.logits),
    )

  def _inverse_cdf(self, quantile):
    """Inverse cumulative density function."""
    probs = jax.nn.softmax(self.logits, axis=-1)
    cum_probs = jnp.cumsum(probs, axis=-1)
    quantile_limits = jnp.concatenate(
        [jnp.zeros_like(cum_probs[..., :1]), cum_probs], axis=-1)
    limits = self._get_category_limits()
    domain_diff = jnp.diff(limits, axis=-1)
    quantile_diff = jnp.diff(quantile_limits, axis=-1)
    slopes = domain_diff / quantile_diff
    quantile_contributions = jnp.minimum(
        quantile_diff,
        jax.nn.relu(quantile[..., None] - quantile_limits[..., :-1]),
    )
    return self.low + jnp.sum(slopes * quantile_contributions, axis=-1)

  @property
  def low(self) -> Array:
    # Broadcasted version of the argument passed in the initializer.
    return jnp.broadcast_to(self._low, self.batch_shape)

  @property
  def high(self) -> Array:
    # Broadcasted version of the argument passed in the initializer.
    return jnp.broadcast_to(self._high, self.batch_shape)

  @property
  def logits(self) -> Array:
    return self._logits

  @property
  def num_bins(self) -> int:
    return self.logits.shape[-1]
