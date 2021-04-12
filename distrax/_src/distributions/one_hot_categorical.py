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
"""OneHotCategorical distribution."""

from typing import Optional, Tuple

import chex
from distrax._src.distributions import categorical
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey


class OneHotCategorical(categorical.Categorical):
  """OneHotCategorical distribution."""

  equiv_tfp_cls = tfd.OneHotCategorical

  def __init__(self,
               logits: Optional[Array] = None,
               probs: Optional[Array] = None,
               dtype: jnp.dtype = jnp.int_):
    """Initializes a OneHotCategorical distribution.

    Args:
      logits: Logit transform of the probability of each category. Only one
        of `logits` or `probs` can be specified.
      probs: Probability of each category. Only one of `logits` or `probs` can
        be specified.
      dtype: The type of event samples.
    """
    super().__init__(logits=logits, probs=probs, dtype=dtype)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return (self.num_categories,)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    new_shape = (n,) + self.logits.shape[:-1]
    draws = jax.random.categorical(
        key=key, logits=self.logits, axis=-1, shape=new_shape)
    draws_one_hot = jax.nn.one_hot(draws, num_classes=self.num_categories)
    return draws_one_hot.astype(self._dtype)

  def log_prob(self, event: Array) -> Array:
    """See `Distribution.log_prob`."""
    return jnp.sum(math.multiply_no_nan(self.logits, event), axis=-1)

  def prob(self, event: Array) -> Array:
    """See `Distribution.prob`."""
    return jnp.sum(math.multiply_no_nan(self.probs, event), axis=-1)

  def mode(self) -> Array:
    """Calculates the mode."""
    preferences = self._probs if self._logits is None else self._logits
    greedy_index = jnp.argmax(preferences, axis=-1)
    return jax.nn.one_hot(greedy_index, self.num_categories).astype(self._dtype)

  def cdf(self, event: Array) -> Array:
    """See `Distribution.cdf`."""
    return jnp.sum(math.multiply_no_nan(
        jnp.cumsum(self.probs, axis=-1), event), axis=-1)
