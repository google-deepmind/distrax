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
"""Dirichlet distribution."""

from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions.beta import Beta
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Dirichlet(distribution.Distribution):
  """Dirichlet distribution with concentration parameter `alpha`.

  The PDF of a Dirichlet distributed random variable `X`, where `X` lives in the
  simplex `(0, 1)^K` with `sum_{k=1}^{K} X_k = 1`, is given by,
  ```
  p(x; alpha) = ( prod_{k=1}^{K} x_k ** (alpha_k - 1) ) / B(alpha)
  ```
  where `B(alpha)` is the multivariate beta function, and the concentration
  parameters `alpha_k > 0`.

  Note that the support of the distribution does not include `x_k = 0` nor
  `x_k = 1`.
  """

  equiv_tfp_cls = tfd.Dirichlet

  def __init__(self, concentration: Array):
    """Initializes a Dirichlet distribution.

    Args:
      concentration: Concentration parameter `alpha` of the distribution. It
        must be an array of length `K >= 2` containing positive values
        (additional dimensions index batches).
    """
    super().__init__()
    self._concentration = conversion.as_float_array(concentration)
    if self._concentration.ndim < 1:
      raise ValueError(
          'The concentration parameter must have at least one dimension.')
    if self._concentration.shape[-1] < 2:
      raise ValueError(
          'The last dimension of the concentration parameter must be '
          'at least 2.')
    self._log_normalization_constant = math.log_beta_multivariate(
        self._concentration)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return self._concentration.shape[-1:]

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._concentration.shape[:-1]

  @property
  def concentration(self) -> Array:
    """Concentration parameter `alpha` of the distribution."""
    return self._concentration

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    out_shape = (n,) + self.batch_shape
    dtype = self._concentration.dtype
    rnd = jax.random.dirichlet(
        key, alpha=self._concentration, shape=out_shape, dtype=dtype)
    return rnd

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    return (jnp.sum((self._concentration - 1.) * jnp.log(value), axis=-1)
            - self._log_normalization_constant)

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    sum_concentration = jnp.sum(self._concentration, axis=-1)
    return (
        self._log_normalization_constant
        + ((sum_concentration - self._concentration.shape[-1])
           * jax.lax.digamma(sum_concentration))
        - jnp.sum((self._concentration - 1.) *
                  jax.lax.digamma(self._concentration), axis=-1)
    )

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._concentration / jnp.sum(
        self._concentration, axis=-1, keepdims=True)

  def mode(self) -> Array:
    """Calculates the mode.

    Returns:
      The mode, an array of shape `batch_shape + event_shape`. If any
      `alpha_k <= 1`, the returned value is `jnp.nan`.
    """
    result_if_valid = (self._concentration - 1.) / jnp.sum(
        self._concentration - 1., axis=-1, keepdims=True)
    return jnp.where(
        jnp.all(self._concentration > 1., axis=-1, keepdims=True),
        result_if_valid,
        jnp.nan)

  def variance(self) -> Array:
    """Calculates the variance."""
    sum_concentration = jnp.sum(self._concentration, axis=-1, keepdims=True)
    norm_concentration = self._concentration / sum_concentration
    return norm_concentration * (1. - norm_concentration) / (
        sum_concentration + 1.)

  def covariance(self) -> Array:
    """Calculates the covariance.

    Returns:
      An array of shape `batch_shape + event_shape + event_shape` with the
      covariance of the distribution.
    """
    sum_concentration = jnp.sum(self._concentration, axis=-1, keepdims=True)
    norm_concentration = self._concentration / sum_concentration
    norm_over_sum = norm_concentration / (sum_concentration + 1.)
    cov = - jnp.expand_dims(norm_over_sum, axis=-1) * jnp.expand_dims(
        norm_concentration, axis=-2)
    cov += jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(norm_over_sum)
    return cov

  def __getitem__(self, index) -> 'Dirichlet':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Dirichlet(concentration=self.concentration[index])


DirichletLike = Union[Beta, tfd.Beta, Dirichlet, tfd.Dirichlet]


def _obtain_concentration(dist: DirichletLike) -> Array:
  """Returns the concentration parameters of the input distribution."""
  if isinstance(dist, Dirichlet):
    concentration = dist.concentration
  elif isinstance(dist, Beta):
    concentration = jnp.stack((dist.alpha, dist.beta), axis=-1)
  elif isinstance(dist, tfd.Beta):
    concentration = jnp.stack(
        (dist.concentration1, dist.concentration0), axis=-1)
  elif isinstance(dist, tfd.Dirichlet):
    concentration = dist.concentration
  return concentration


def _kl_divergence_dirichlet_dirichlet(
    dist1: DirichletLike,
    dist2: DirichletLike,
    *unused_args,
    **unused_kwargs,
) -> Array:
  """KL divergence KL(dist1 || dist2) between two Dirichlet distributions.

  Args:
    dist1: A Dirichlet or Beta distribution.
    dist2: A Dirichlet or Beta distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  concentration1 = _obtain_concentration(dist1)
  concentration2 = _obtain_concentration(dist2)
  if concentration1.shape[-1] != concentration2.shape[-1]:
    raise ValueError(
        f'The two distributions must have the same event dimension, but got '
        f'{concentration1.shape[-1]} and {concentration2.shape[-1]} '
        f'dimensions.')
  sum_concentration1 = jnp.sum(concentration1, axis=-1, keepdims=True)
  t1 = (math.log_beta_multivariate(concentration2)
        - math.log_beta_multivariate(concentration1))
  t2 = jnp.sum((concentration1 - concentration2) * (
      jax.lax.digamma(concentration1) - jax.lax.digamma(sum_concentration1)),
               axis=-1)
  return t1 + t2


# Register the KL functions with TFP.
tfd.RegisterKL(Dirichlet, Dirichlet)(_kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(Dirichlet, Dirichlet.equiv_tfp_cls)(
    _kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(Dirichlet.equiv_tfp_cls, Dirichlet)(
    _kl_divergence_dirichlet_dirichlet)

tfd.RegisterKL(Dirichlet, Beta)(_kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(Beta, Dirichlet)(_kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(Dirichlet, tfd.Beta)(_kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(tfd.Beta, Dirichlet)(_kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(tfd.Dirichlet, Beta)(_kl_divergence_dirichlet_dirichlet)
tfd.RegisterKL(Beta, tfd.Dirichlet)(_kl_divergence_dirichlet_dirichlet)
