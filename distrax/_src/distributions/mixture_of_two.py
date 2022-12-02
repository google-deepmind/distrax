# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""A simple mixture of two (possibly heterogeneous) distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution as base_distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
DistributionLike = base_distribution.DistributionLike


class MixtureOfTwo(base_distribution.Distribution):
  """A mixture of two distributions."""

  def __init__(
      self,
      prob_a: Numeric,
      component_a: DistributionLike,
      component_b: DistributionLike):
    """Creates a mixture of two distributions.

    Differently from `MixtureSameFamily` the component distributions are allowed
    to belong to different families.

    Args:
      prob_a: a scalar weight for the `component_a`, is a float or a rank 0
        vector.
      component_a: the first component distribution.
      component_b: the second component distribution.
    """
    super().__init__()
    # Validate inputs.
    chex.assert_rank(prob_a, 0)
    if component_a.event_shape != component_b.event_shape:
      raise ValueError(
          f'The component distributions must have the same event shape, but '
          f'{component_a.event_shape} != {component_b.event_shape}.')
    if component_a.batch_shape != component_b.batch_shape:
      raise ValueError(
          f'The component distributions must have the same batch shape, but '
          f'{component_a.batch_shape} != {component_b.batch_shape}.')
    if component_a.dtype != component_b.dtype:
      raise ValueError(
          'The component distributions must have the same dtype, but'
          f' {component_a.dtype} != {component_b.dtype}.')
    # Store args.
    self._prob_a = prob_a
    self._component_a = conversion.as_distribution(component_a)
    self._component_b = conversion.as_distribution(component_b)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key, key_a, key_b, mask_key = jax.random.split(key, num=4)
    mask_from_a = jax.random.bernoulli(mask_key, p=self._prob_a, shape=[n])
    sample_a = self._component_a.sample(seed=key_a, sample_shape=n)
    sample_b = self._component_b.sample(seed=key_b, sample_shape=n)
    mask_from_a = jnp.expand_dims(mask_from_a, tuple(range(1, sample_a.ndim)))
    return jnp.where(mask_from_a, sample_a, sample_b)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    logp1 = jnp.log(self._prob_a) + self._component_a.log_prob(value)
    logp2 = jnp.log(1 - self._prob_a) + self._component_b.log_prob(value)
    return jnp.logaddexp(logp1, logp2)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    return self._component_a.event_shape

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return self._component_a.batch_shape

  @property
  def prob_a(self) -> Numeric:
    return self._prob_a

  @property
  def prob_b(self) -> Numeric:
    return 1. - self._prob_a

  def __getitem__(self, index) -> 'MixtureOfTwo':
    """See `Distribution.__getitem__`."""
    index = base_distribution.to_batch_shape_index(self.batch_shape, index)
    return MixtureOfTwo(
        prob_a=self.prob_a,
        component_a=self._component_a[index],
        component_b=self._component_b[index])
