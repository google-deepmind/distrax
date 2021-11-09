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
"""Mixture distributions."""

from typing import Tuple

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey
DistributionLike = distribution.DistributionLike
CategoricalLike = categorical.CategoricalLike


class MixtureSameFamily(distribution.Distribution):
  """Mixture with components provided from a single batched distribution."""

  equiv_tfp_cls = tfd.MixtureSameFamily

  def __init__(self,
               mixture_distribution: CategoricalLike,
               components_distribution: DistributionLike):
    """Initializes a mixture distribution for components of a shared family.

    Args:
      mixture_distribution: Distribution over selecting components.
      components_distribution: Component distribution, with rightmost batch
        dimension indexing components.
    """
    super().__init__()
    mixture_distribution = conversion.as_distribution(mixture_distribution)
    components_distribution = conversion.as_distribution(
        components_distribution)
    self._mixture_distribution = mixture_distribution
    self._components_distribution = components_distribution

    # Store normalized weights (last axis of logits is for components).
    # This uses the TFP API, which is replicated in Distrax.
    self._mixture_log_probs = jax.nn.log_softmax(
        mixture_distribution.logits_parameter(), axis=-1)

    batch_shape_mixture = mixture_distribution.batch_shape
    batch_shape_components = components_distribution.batch_shape
    if batch_shape_mixture != batch_shape_components[:-1]:
      msg = (f'`mixture_distribution.batch_shape` '
             f'({mixture_distribution.batch_shape}) is not compatible with '
             f'`components_distribution.batch_shape` '
             f'({components_distribution.batch_shape}`)')
      raise ValueError(msg)

  @property
  def components_distribution(self):
    """The components distribution."""
    return self._components_distribution

  @property
  def mixture_distribution(self):
    """The mixture distribution."""
    return self._mixture_distribution

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return self._components_distribution.event_shape

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._components_distribution.batch_shape[:-1]

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key_mix, key_components = jax.random.split(key)
    mix_sample = self.mixture_distribution.sample(sample_shape=n, seed=key_mix)

    num_components = self._components_distribution.batch_shape[-1]

    # Sample from all components, then multiply with a one-hot mask and sum.
    # While this does computation that is not used eventually, it is faster on
    # GPU/TPUs, which excel at batched operations (as opposed to indexing). It
    # is in particular more efficient than using `gather` or `where` operations.
    mask = jax.nn.one_hot(mix_sample, num_components)
    samples_all = self.components_distribution.sample(sample_shape=n,
                                                      seed=key_components)

    # Make mask broadcast with (potentially multivariate) samples.
    mask = mask.reshape(mask.shape + (1,) * len(self.event_shape))

    # Need to sum over the component axis, which is the last one for scalar
    # components, the second-last one for 1-dim events, etc.
    samples = jnp.sum(samples_all * mask, axis=-1 - len(self.event_shape))
    return samples

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    # Add component axis to make input broadcast with components distribution.
    expanded = jnp.expand_dims(value, axis=-1 - len(self.event_shape))
    # Compute `log_prob` in every component.
    lp = self.components_distribution.log_prob(expanded)
    # Last batch axis is number of components, i.e. last axis of `lp` below.
    # Last axis of mixture log probs are components, so reduce last axis.
    return jax.scipy.special.logsumexp(a=lp + self._mixture_log_probs, axis=-1)

  def mean(self) -> Array:
    """Calculates the mean."""
    means = self.components_distribution.mean()
    weights = jnp.exp(self._mixture_log_probs)
    # Broadcast weights over event shape, and average over component axis.
    weights = weights.reshape(weights.shape + (1,) * len(self.event_shape))
    return jnp.sum(means * weights, axis=-1 - len(self.event_shape))

  def variance(self) -> Array:
    """Calculates the variance."""
    means = self.components_distribution.mean()
    variances = self.components_distribution.variance()
    weights = jnp.exp(self._mixture_log_probs)
    # Make weights broadcast over event shape.
    weights = weights.reshape(weights.shape + (1,) * len(self.event_shape))
    # Component axis to reduce over.
    component_axis = -1 - len(self.event_shape)

    # Using: Var(Y) = E[Var(Y|X)] + Var(E[Y|X]).
    mean = jnp.sum(means * weights, axis=component_axis)
    mean_cond_var = jnp.sum(weights * variances, axis=component_axis)
    # Need to add an axis to `mean` to make it broadcast over components.
    sq_diff = jnp.square(means - jnp.expand_dims(mean, axis=component_axis))
    var_cond_mean = jnp.sum(weights * sq_diff, axis=component_axis)
    return mean_cond_var + var_cond_mean

  def __getitem__(self, index) -> 'MixtureSameFamily':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return MixtureSameFamily(
        mixture_distribution=self.mixture_distribution[index],
        components_distribution=self.components_distribution[index])
