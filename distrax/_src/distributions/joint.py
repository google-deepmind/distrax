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

from typing import Any, Callable, Tuple, TypeVar

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


DistributionT = TypeVar(
    'DistributionT', bound=distribution.NestedT[conversion.DistributionLike]
)


def _is_leaf_distribution(x: Any) -> bool:
  return isinstance(x, conversion.DistributionLike)


def _map_up_to_distribution(
    fn: Callable[..., Any], distributions: Any, *xs: Any
) -> Any:
  """Maps `fn` over `x` up to `tree` to determine leaves."""
  return jax.tree.map(fn, distributions, *xs, is_leaf=_is_leaf_distribution)


def _leaves_up_to_distribution(distributions: Any) -> Any:
  """Flattens `distributions` up to `tree` to determine leaves."""
  return jax.tree.leaves(distributions, is_leaf=_is_leaf_distribution)


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
    self._distributions: DistributionT = _map_up_to_distribution(
        conversion.as_distribution, distributions
    )
    self._distributions_treedef = jax.tree.structure(
        self._distributions,
        is_leaf=_is_leaf_distribution,
    )
    self._distributions_leaves = _leaves_up_to_distribution(self._distributions)

    batch_shape = None
    for path, dist in jax.tree.leaves_with_path(
        self._distributions,
        is_leaf=_is_leaf_distribution,
        is_leaf_takes_path=False,
    ):
      batch_shape = batch_shape or dist.batch_shape
      first_path = '.'.join(map(str, path))
      if dist.batch_shape != batch_shape:
        path = '.'.join(map(str, path))
        raise ValueError(
            'Joint distributions must have the same batch shape, but '
            f'distribution "{dist.name}" at location {path} had batch shape '
            f'{dist.batch_shape} which is not equal to the batch shape '
            f'{batch_shape} of the distribution at location {first_path}.'
        )

  def _sample_n(self, key: chex.PRNGKey, n: int) -> distribution.EventT:
    keys = list(jax.random.split(key, len(self._distributions_leaves)))
    keys = self._distributions_treedef.unflatten(keys)
    return _map_up_to_distribution(
        lambda d, k: d.sample(seed=k, sample_shape=n), self._distributions, keys
    )

  def _sample_n_and_log_prob(
      self, key: chex.PRNGKey, n: int
  ) -> Tuple[distribution.EventT, chex.Array]:
    keys = list(jax.random.split(key, len(self._distributions_leaves)))
    keys = self._distributions_treedef.unflatten(keys)
    samples_and_log_probs = _map_up_to_distribution(
        lambda d, k: d.sample_and_log_prob(seed=k, sample_shape=n),
        self._distributions,
        keys,
    )
    samples = _map_up_to_distribution(
        lambda _, p: p[0], self._distributions, samples_and_log_probs
    )
    log_probs = _map_up_to_distribution(
        lambda _, p: p[1], self._distributions, samples_and_log_probs
    )
    log_probs = jnp.stack(jax.tree.leaves(log_probs))
    log_probs = jnp.sum(log_probs, axis=0)
    return samples, log_probs

  def log_prob(self, value: distribution.EventT) -> chex.Array:
    """Compute the total log probability of the distributions in the tree."""
    log_probs = _map_up_to_distribution(
        lambda dist, value: dist.log_prob(value), self._distributions, value
    )
    log_probs = jnp.stack(jax.tree.leaves(log_probs))
    return jnp.sum(log_probs, axis=0)

  @property
  def distributions(self) -> DistributionT:
    return self._distributions

  @property
  def event_shape(self) -> distribution.ShapeT:
    return _map_up_to_distribution(
        lambda dist: dist.event_shape, self._distributions
    )

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    return self._distributions_leaves[0].batch_shape

  @property
  def dtype(self) -> distribution.DTypeT:
    return _map_up_to_distribution(lambda dist: dist.dtype, self._distributions)

  def entropy(self) -> chex.Array:
    return sum(dist.entropy() for dist in self._distributions_leaves)

  def log_cdf(self, value: distribution.EventT) -> chex.Array:
    return sum(
        dist.log_cdf(v)
        for dist, v in zip(self._distributions_leaves, jax.tree.leaves(value))
    )

  def mean(self) -> distribution.EventT:
    """Calculates the mean."""
    return _map_up_to_distribution(
        lambda dist: dist.mean(), self._distributions
    )

  def median(self) -> distribution.EventT:
    """Calculates the median."""
    return _map_up_to_distribution(
        lambda dist: dist.median(), self._distributions
    )

  def mode(self) -> distribution.EventT:
    """Calculates the mode."""
    return _map_up_to_distribution(
        lambda dist: dist.mode(), self._distributions
    )

  def __getitem__(self, index) -> 'Joint':
    """See `Distribution.__getitem__`."""
    return Joint(
        _map_up_to_distribution(lambda dist: dist[index], self._distributions)
    )


def _kl_divergence_joint_joint(
    dist1: Joint, dist2: Joint, *unused_args, **unused_kwargs
) -> chex.Array:
  """Calculates the KL divergence between two Joint distributions."""
  treedef1 = jax.tree.structure(
      dist1.distributions, is_leaf=_is_leaf_distribution
  )
  treedef2 = jax.tree.structure(
      dist2.distributions, is_leaf=_is_leaf_distribution
  )
  if treedef1 != treedef2:
    raise ValueError(
        'Joint distributions must have the same tree structure, but\n'
        f'{treedef1=}\n{treedef2=}.'
    )
  return sum(
      inner1.kl_divergence(inner2)
      for inner1, inner2 in zip(
          _leaves_up_to_distribution(dist1.distributions),
          _leaves_up_to_distribution(dist2.distributions),
      )
  )


tfp.distributions.RegisterKL(Joint, Joint)(_kl_divergence_joint_joint)
