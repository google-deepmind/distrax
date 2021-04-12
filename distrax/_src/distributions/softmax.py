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
import jax.numpy as jnp


Array = chex.Array


class Softmax(categorical.Categorical):
  """Categorical implementing a softmax over logits, with given temperature.

  Given a set of logits, the probability mass is distributed such that each
  index `i` has probability `exp(logits[i]/τ)/Σ(exp(logits/τ)` where τ is a
  scalar `temperature` parameter such that for τ→0, the distribution
  becomes fully greedy, and for τ→0 the distribution becomes fully uniform.
  """

  def __init__(self,
               logits: Array,
               temperature: float = 1.,
               dtype: jnp.dtype = jnp.int_):
    """Initializes a Softmax distribution.

    Args:
      logits: Logit transform of the probability of each category.
      temperature: Softmax temperature τ.
      dtype: The type of event samples.
    """
    scaled_logits = logits / temperature
    super().__init__(logits=scaled_logits, dtype=dtype)

