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
"""Distrax joint distribution over a tree of distributions."""

from typing import Tuple, TypeVar

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import tree

tfd = tfp.distributions


DistributionT = TypeVar(
    'DistributionT', bound=distribution.NestedT[conversion.DistributionLike])


class Joint(distribution.Distribution):
  """Joint distribution over a tree of statistically independent distributions.

  Samples from the Joint distribution take the form of a tree structure that
  matches the structure of the underlying distributions. Log-probabilities
  are summed over the tree.

  All distributions in the tree must have the same `batch_shape` in order for
  log-probabilities to be computed correctly and for the `batch_shape` of the
  Joint distribution to be correct.
  """

  def __init__(self, distributions: DistributionT):
    """Initializes a Joint distribution over a tree of distributions.

    Args:
      distributions: Tree of distributions that must have the same batch shape.
    """
    super().__init__()
    self._distributions = tree.map_structure(conversion.as_distribution,
                                             distributions)
    batch_shape = None
    first_path = None
    for path, dist in tree.flatten_with_path(self._distributions):
      batch_shape = batch_shape or dist.batch_shape
      first_path = '.'.join(map(str, path))
      if dist.batch_shape != batch_shape:
        path = '.'.join(map(str, path))
        raise ValueError(
            f'Joint distributions must have the same batch shape, but '
            f'distribution "{dist.name}" at location {path} had batch shape '
            f'{dist.batch_shape} which is not equal to the batch shape '
            f'{batch_shape} of the distribution at location {first_path}.')

  def _sample_n(
      self,
      key: chex.PRNGKey,
      n: int) -> distribution.EventT:
    keys = list(jax.random.split(key, len(tree.flatten(self._distributions))))
    keys = tree.unflatten_as(self._distributions, keys)
    return tree.map_structure(lambda d, k: d.sample(seed=k, sample_shape=n),
                              self._distributions, keys)

  def _sample_n_and_log_prob(
      self,
      key: chex.PRNGKey,
      n: int) -> Tuple[distribution.EventT, chex.Array]:
    keys = list(jax.random.split(key, len(tree.flatten(self._distributions))))
    keys = tree.unflatten_as(self._distributions, keys)
    samples_and_log_probs = tree.map_structure(
        lambda d, k: d.sample_and_log_prob(seed=k, sample_shape=n),
        self._distributions, keys)
    samples = tree.map_structure_up_to(
        self._distributions, lambda p: p[0], samples_and_log_probs)
    log_probs = tree.map_structure_up_to(
        self._distributions, lambda p: p[1], samples_and_log_probs)
    log_probs = jnp.stack(tree.flatten(log_probs))
    log_probs = jnp.sum(log_probs, axis=0)
    return samples, log_probs

  def log_prob(self, value: distribution.EventT) -> chex.Array:
    """Compute the total log probability of the distributions in the tree."""
    log_probs = tree.map_structure(lambda dist, value: dist.log_prob(value),
                                   self._distributions, value)
    log_probs = jnp.stack(tree.flatten(log_probs))
    return jnp.sum(log_probs, axis=0)

  @property
  def distributions(self) -> DistributionT:
    return self._distributions

  @property
  def event_shape(self) -> distribution.ShapeT:
    return tree.map_structure(lambda dist: dist.event_shape,
                              self._distributions)

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return tree.flatten(self._distributions)[0].batch_shape

  @property
  def dtype(self) -> distribution.DTypeT:
    return tree.map_structure(lambda dist: dist.dtype, self._distributions)

  def entropy(self) -> chex.Array:
    return sum(dist.entropy() for dist in tree.flatten(self._distributions))

  def log_cdf(self, value: distribution.EventT) -> chex.Array:
    return sum(dist.log_cdf(v)
               for dist, v in zip(tree.flatten(self._distributions),
                                  tree.flatten(value)))

  def mean(self) -> distribution.EventT:
    """Calculates the mean."""
    return tree.map_structure(lambda dist: dist.mean(), self._distributions)

  def median(self) -> distribution.EventT:
    """Calculates the median."""
    return tree.map_structure(lambda dist: dist.median(), self._distributions)

  def mode(self) -> distribution.EventT:
    """Calculates the mode."""
    return tree.map_structure(lambda dist: dist.mode(), self._distributions)

  def __getitem__(self, index) -> 'Joint':
    """See `Distribution.__getitem__`."""
    return Joint(tree.map_structure(lambda dist: dist[index],
                                    self._distributions))


def _kl_divergence_joint_joint(
    dist1: Joint, dist2: Joint, *unused_args, **unused_kwargs) -> chex.Array:
  tree.assert_same_structure(
      dist1.distributions, dist2.distributions, check_types=False)
  return sum(inner1.kl_divergence(inner2)
             for inner1, inner2 in zip(tree.flatten(dist1.distributions),
                                       tree.flatten(dist2.distributions)))

tfd.RegisterKL(Joint, Joint)(_kl_divergence_joint_joint)
