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
"""Independent distribution."""

from typing import Callable, Optional, Tuple, Union

import chex
from distrax._src.distributions import distribution as distrax_distribution
from distrax._src.utils import conversion
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey
DistributionLike = distrax_distribution.DistributionLike


class Independent(distrax_distribution.Distribution):
  """Independent distribution obtained from child distributions."""

  equiv_tfp_cls = tfd.Independent

  def __init__(self,
               distribution: DistributionLike,
               reinterpreted_batch_ndims: Optional[int] = None):
    """Initializes an Independent distribution.

    Args:
      distribution: Base distribution instance.
      reinterpreted_batch_ndims: Number of event dimensions.
    """
    super().__init__()
    distribution = conversion.as_distribution(distribution)
    self._distribution = distribution

    # Check if event shape is a tuple of integers (i.e. not nested).
    event_shape = distribution.event_shape
    if not (isinstance(event_shape, tuple) and
            all(isinstance(i, int) for i in event_shape)):
      raise ValueError(
          f"'Independent' currently only supports distributions with Array "
          f"events (i.e. not nested). Received '{distribution.name}' with "
          f"event shape '{distribution.event_shape}'.")

    dist_batch_shape = distribution.batch_shape
    if reinterpreted_batch_ndims is not None:
      dist_batch_ndims = len(dist_batch_shape)
      if reinterpreted_batch_ndims > dist_batch_ndims:
        raise ValueError(
            f'`reinterpreted_batch_ndims` is {reinterpreted_batch_ndims}, but'
            f' distribution `{distribution.name}` has only {dist_batch_ndims}'
            f' batch dimensions.')
      elif reinterpreted_batch_ndims < 0:
        raise ValueError(f'`reinterpreted_batch_ndims` can\'t be negative; got'
                         f' {reinterpreted_batch_ndims}.')
      self._reinterpreted_batch_ndims = reinterpreted_batch_ndims
    else:
      self._reinterpreted_batch_ndims = max(len(dist_batch_shape) - 1, 0)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    dist_batch_shape = self._distribution.batch_shape
    event_ndims = len(dist_batch_shape) - self._reinterpreted_batch_ndims
    return dist_batch_shape[event_ndims:] + self._distribution.event_shape

  @property
  def distribution(self):
    return self._distribution

  @property
  def reinterpreted_batch_ndims(self) -> int:
    return self._reinterpreted_batch_ndims

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    dist_batch_shape = self._distribution.batch_shape
    d = len(dist_batch_shape) - self.reinterpreted_batch_ndims
    return dist_batch_shape[:d]

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    return self._distribution.sample(seed=key, sample_shape=n)

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    samples, log_prob = self._distribution.sample_and_log_prob(
        seed=key, sample_shape=n)
    log_prob = self._reduce(jnp.sum, log_prob)
    return samples, log_prob

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    return self._reduce(jnp.sum, self._distribution.log_prob(value))

  def entropy(self) -> Array:
    """See `Distribution.entropy`."""
    return self._reduce(jnp.sum, self._distribution.entropy())

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return self._reduce(jnp.sum, self._distribution.log_cdf(value))

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._distribution.mean()

  def median(self) -> Array:
    """Calculates the median."""
    return self._distribution.median()

  def variance(self) -> Array:
    """Calculates the variance."""
    return self._distribution.variance()

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self._distribution.stddev()

  def mode(self) -> Array:
    """Calculates the mode."""
    return self._distribution.mode()

  def _reduce(self, fn: Callable[..., Array], value: Array) -> Array:
    return fn(value,
              axis=[-i - 1 for i in range(0, self.reinterpreted_batch_ndims)])

  def __getitem__(self, index) -> 'Independent':
    """See `Distribution.__getitem__`."""
    index = distrax_distribution.to_batch_shape_index(self.batch_shape, index)
    return Independent(
        distribution=self.distribution[index],
        reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)


def _kl_divergence_independent_independent(
    dist1: Union[Independent, tfd.Independent],
    dist2: Union[Independent, tfd.Independent],
    *args, **kwargs) -> Array:
  """Batched KL divergence `KL(dist1 || dist2)` for Independent distributions.

  Args:
    dist1: instance of an Independent distribution.
    dist2: instance of an Independent distribution.
    *args: Additional args.
    **kwargs: Additional kwargs.

  Returns:
    Batchwise `KL(dist1 || dist2)`
  """
  p = dist1.distribution
  q = dist2.distribution

  if dist1.event_shape == dist2.event_shape:
    if p.event_shape == q.event_shape:
      num_reduce_dims = len(dist1.event_shape) - len(p.event_shape)
      reduce_dims = [-i - 1 for i in range(0, num_reduce_dims)]
      kl_divergence = jnp.sum(p.kl_divergence(q, *args, **kwargs),
                              axis=reduce_dims)
    else:
      raise NotImplementedError(
          f'KL between Independents whose inner distributions have different '
          f'event shapes is not supported: obtained {p.event_shape} and '
          f'{q.event_shape}.')
  else:
    raise ValueError(f'Event shapes {dist1.event_shape} and {dist2.event_shape}'
                     f' do not match.')

  return kl_divergence


# Register the KL functions with TFP.
tfd.RegisterKL(Independent, Independent)(_kl_divergence_independent_independent)
tfd.RegisterKL(Independent, Independent.equiv_tfp_cls)(
    _kl_divergence_independent_independent)
tfd.RegisterKL(Independent.equiv_tfp_cls, Independent)(
    _kl_divergence_independent_independent)
