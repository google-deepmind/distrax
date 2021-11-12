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
"""Deterministic distribution."""

from typing import Optional, Tuple, Union

import chex
from distrax._src.distributions import distribution
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Deterministic(distribution.Distribution):
  """Scalar Deterministic distribution on the real line."""

  equiv_tfp_cls = tfd.Deterministic

  def __init__(self,
               loc: Numeric,
               atol: Optional[Numeric] = None,
               rtol: Optional[Numeric] = None):
    """Initializes a Deterministic distribution.

    Args:
      loc: Batch of points on which the distribution is supported.
      atol: Absolute tolerance for comparing closeness to `loc`. It must be
        broadcastable with `loc`, and it must not lead to additional batch
        dimensions after broadcasting.
      rtol: Relative tolerance for comparing closeness to `loc`. It must be
        broadcastable with `loc`, and it must not lead to additional batch
        dimensions after broadcasting.
    """
    super().__init__()
    self._loc = jnp.asarray(loc)
    self._atol = jnp.asarray(0. if atol is None else atol)
    self._rtol = jnp.asarray(0. if rtol is None else rtol)
    if len(self._rtol.shape) > len(self._loc.shape):
      raise ValueError(f'The parameter `rtol` cannot have more dimensions than '
                       f'`loc`, but their shapes are {self._rtol.shape} and '
                       f'{self._loc.shape}, respectively.')
    if len(self._atol.shape) > len(self._loc.shape):
      raise ValueError(f'The parameter `atol` cannot have more dimensions than '
                       f'`loc`, but their shapes are {self._atol.shape} and '
                       f'{self._loc.shape}, respectively.')

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of the events."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._loc.shape

  @property
  def loc(self) -> Array:
    """Point(s) on which this distribution is supported."""
    return self._loc

  @property
  def atol(self) -> Array:
    """Absolute tolerance for comparing closeness to `loc`."""
    return jnp.broadcast_to(self._atol, self.batch_shape)

  @property
  def rtol(self) -> Array:
    """Relative tolerance for comparing closeness to `loc`."""
    return jnp.broadcast_to(self._rtol, self.batch_shape)

  @property
  def slack(self) -> Array:
    return jnp.where(
        self.rtol == 0,
        self.atol,
        self.atol + self.rtol * jnp.abs(self.loc))

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    del key  # unused
    loc = jnp.expand_dims(self.loc, axis=0)
    return jnp.repeat(loc, n, axis=0)

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    samples = self._sample_n(key, n)
    log_prob = jnp.zeros_like(samples)
    return samples, log_prob

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    return jnp.log(self.prob(value))

  def prob(self, value: Array) -> Array:
    """See `Distribution.prob`."""
    return jnp.where(
        jnp.abs(value - self.loc) <= self.slack, 1., 0.)

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    return jnp.zeros(self.batch_shape, jnp.float_)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.zeros(self.batch_shape, jnp.float_)

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self.variance()

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jnp.where(value >= self.loc - self.slack, 1., 0.)

  def __getitem__(self, index) -> 'Deterministic':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Deterministic(
        loc=self.loc[index], atol=self.atol[index], rtol=self.rtol[index])


def _kl_divergence_deterministic_deterministic(
    dist1: Union[Deterministic, tfd.Deterministic],
    dist2: Union[Deterministic, tfd.Deterministic],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """KL divergence `KL(dist1 || dist2)` between two Deterministic distributions.

  Note that the KL divergence is infinite if the support of `dist1` is not a
  subset of the support of `dist2`.

  Args:
    dist1: A Deterministic distribution.
    dist2: A Deterministic distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  slack2 = dist2.atol + dist2.rtol * jnp.abs(dist2.loc)
  return - jnp.log(jnp.where(jnp.abs(dist1.loc - dist2.loc) <= slack2, 1., 0.))


# Register the KL functions with TFP.
tfd.RegisterKL(Deterministic, Deterministic)(
    _kl_divergence_deterministic_deterministic)
tfd.RegisterKL(Deterministic, Deterministic.equiv_tfp_cls)(
    _kl_divergence_deterministic_deterministic)
tfd.RegisterKL(Deterministic.equiv_tfp_cls, Deterministic)(
    _kl_divergence_deterministic_deterministic)
