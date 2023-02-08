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
"""Greedy distributions with respect to a set of preferences."""

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
import jax.numpy as jnp


Array = chex.Array


def _argmax_with_random_tie_breaking(preferences: Array) -> Array:
  """Compute probabilities greedily with respect to a set of preferences."""
  optimal_actions = preferences == preferences.max(axis=-1, keepdims=True)
  return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)


class Greedy(categorical.Categorical):
  """A Categorical distribution that is greedy with respect to some preferences.

  Given a set of unnormalized preferences, the probability mass is distributed
  equally among all indices `i` such that `preferences[i] = max(preferences)`,
  all other indices will be assigned a probability of zero.
  """

  def __init__(self, preferences: Array, dtype: jnp.dtype = int):
    """Initializes a Greedy distribution.

    Args:
      preferences: Unnormalized preferences.
      dtype: The type of event samples.
    """
    self._preferences = jnp.asarray(preferences)
    probs = _argmax_with_random_tie_breaking(self._preferences)
    super().__init__(probs=probs, dtype=dtype)

  @property
  def preferences(self) -> Array:
    """Unnormalized preferences."""
    return self._preferences

  def __getitem__(self, index) -> 'Greedy':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Greedy(
        preferences=self.preferences[index], dtype=self.dtype)
