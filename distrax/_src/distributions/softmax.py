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
"""Softmax distribution."""

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
import jax.numpy as jnp


Array = chex.Array


class Softmax(categorical.Categorical):
  """Categorical implementing a softmax over logits, with given temperature.

  Given a set of logits, the probability mass is distributed such that each
  index `i` has probability `exp(logits[i]/τ)/Σ(exp(logits/τ)` where τ is a
  scalar `temperature` parameter such that for τ→0, the distribution
  becomes fully greedy, and for τ→∞ the distribution becomes fully uniform.
  """

  def __init__(self,
               logits: Array,
               temperature: float = 1.,
               dtype: jnp.dtype = int):
    """Initializes a Softmax distribution.

    Args:
      logits: Logit transform of the probability of each category.
      temperature: Softmax temperature τ.
      dtype: The type of event samples.
    """
    self._temperature = temperature
    self._unscaled_logits = logits
    scaled_logits = logits / temperature
    super().__init__(logits=scaled_logits, dtype=dtype)

  @property
  def temperature(self) -> float:
    """The softmax temperature parameter."""
    return self._temperature

  @property
  def unscaled_logits(self) -> Array:
    """The logits of the distribution before the temperature scaling."""
    return self._unscaled_logits

  def __getitem__(self, index) -> 'Softmax':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Softmax(
        logits=self.unscaled_logits[index],
        temperature=self.temperature,
        dtype=self.dtype)
