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
"""Epsilon-Greedy distributions with respect to a set of preferences."""

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
import jax.numpy as jnp


Array = chex.Array


def _argmax_with_random_tie_breaking(preferences: Array) -> Array:
  """Compute probabilities greedily with respect to a set of preferences."""
  optimal_actions = preferences == preferences.max(axis=-1, keepdims=True)
  return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)


def _mix_probs_with_uniform(probs: Array, epsilon: float) -> Array:
  """Mix an arbitrary categorical distribution with a uniform distribution."""
  num_actions = probs.shape[-1]
  uniform_probs = jnp.ones_like(probs) / num_actions
  return (1 - epsilon) * probs + epsilon * uniform_probs


class EpsilonGreedy(categorical.Categorical):
  """A Categorical that is ε-greedy with respect to some preferences.

  Given a set of unnormalized preferences, the distribution is a mixture
  of the Greedy and Uniform distribution; with weight (1-ε) and ε, respectively.
  """

  def __init__(self,
               preferences: Array,
               epsilon: float,
               dtype: jnp.dtype = int):
    """Initializes an EpsilonGreedy distribution.

    Args:
      preferences: Unnormalized preferences.
      epsilon: Mixing parameter ε.
      dtype: The type of event samples.
    """
    self._preferences = jnp.asarray(preferences)
    self._epsilon = epsilon
    greedy_probs = _argmax_with_random_tie_breaking(self._preferences)
    probs = _mix_probs_with_uniform(greedy_probs, epsilon)
    super().__init__(probs=probs, dtype=dtype)

  @property
  def epsilon(self) -> float:
    """Mixing parameters of the distribution."""
    return self._epsilon

  @property
  def preferences(self) -> Array:
    """Unnormalized preferences."""
    return self._preferences

  def __getitem__(self, index) -> 'EpsilonGreedy':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return EpsilonGreedy(
        preferences=self.preferences[index],
        epsilon=self.epsilon,
        dtype=self.dtype)
