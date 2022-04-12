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
"""Beta distribution."""

from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Beta(distribution.Distribution):
  """Beta distribution with parameters `alpha` and `beta`.

  The PDF of a Beta distributed random variable `X` is defined on the interval
  `0 <= X <= 1` and has the form:
  ```
  p(x; alpha, beta) = x ** {alpha - 1} * (1 - x) ** (beta - 1) / B(alpha, beta)
  ```
  where `B(alpha, beta)` is the beta function, and the `alpha, beta > 0` are the
  shape parameters.

  Note that the support of the distribution does not include `x = 0` or `x = 1`
  if `alpha < 1` or `beta < 1`, respectively.
  """

  equiv_tfp_cls = tfd.Beta

  def __init__(self, alpha: Numeric, beta: Numeric):
    """Initializes a Beta distribution.

    Args:
      alpha: Shape parameter `alpha` of the distribution. Must be positive.
      beta: Shape parameter `beta` of the distribution. Must be positive.
    """
    super().__init__()
    self._alpha = conversion.as_float_array(alpha)
    self._beta = conversion.as_float_array(beta)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._alpha.shape, self._beta.shape)
    self._log_normalization_constant = math.log_beta(self._alpha, self._beta)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def alpha(self) -> Array:
    """Shape parameter `alpha` of the distribution."""
    return jnp.broadcast_to(self._alpha, self.batch_shape)

  @property
  def beta(self) -> Array:
    """Shape parameter `beta` of the distribution."""
    return jnp.broadcast_to(self._beta, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._alpha, self._beta)
    rnd = jax.random.beta(
        key, a=self._alpha, b=self._beta, shape=out_shape, dtype=dtype)
    return rnd

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    result = ((self._alpha - 1.) * jnp.log(value)
              + (self._beta - 1.) * jnp.log(1. - value)
              - self._log_normalization_constant)
    return jnp.where(
        jnp.logical_or(jnp.logical_and(self._alpha == 1., value == 0.),
                       jnp.logical_and(self._beta == 1., value == 1.)),
        -self._log_normalization_constant,
        result
    )

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jax.scipy.special.betainc(self._alpha, self._beta, value)

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    return (
        self._log_normalization_constant
        - (self._alpha - 1.) * jax.lax.digamma(self._alpha)
        - (self._beta - 1.) * jax.lax.digamma(self._beta)
        + (self._alpha + self._beta - 2.) * jax.lax.digamma(
            self._alpha + self._beta)
    )

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._alpha / (self._alpha + self._beta)

  def variance(self) -> Array:
    """Calculates the variance."""
    sum_alpha_beta = self._alpha + self._beta
    return self._alpha * self._beta / (
        jnp.square(sum_alpha_beta) * (sum_alpha_beta + 1.))

  def mode(self) -> Array:
    """Calculates the mode.

    Returns:
      The mode, an array of shape `batch_shape`. The mode is not defined if
      `alpha = beta = 1`, or if `alpha < 1` and `beta < 1`. For these cases,
      the returned value is `jnp.nan`.
    """
    return jnp.where(
        jnp.logical_and(self._alpha > 1., self._beta > 1.),
        (self._alpha - 1.) / (self._alpha + self._beta - 2.),
        jnp.where(
            jnp.logical_and(self._alpha <= 1., self._beta > 1.),
            0.,
            jnp.where(
                jnp.logical_and(self._alpha > 1., self._beta <= 1.),
                1., jnp.nan)))

  def __getitem__(self, index) -> 'Beta':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Beta(alpha=self.alpha[index], beta=self.beta[index])


BetaLike = Union[Beta, tfd.Beta]


def _obtain_alpha_beta(dist: BetaLike) -> Tuple[Array, Array]:
  if isinstance(dist, Beta):
    alpha, beta = dist.alpha, dist.beta
  elif isinstance(dist, tfd.Beta):
    alpha, beta = dist.concentration1, dist.concentration0
  return alpha, beta


def _kl_divergence_beta_beta(
    dist1: BetaLike,
    dist2: BetaLike,
    *unused_args,
    **unused_kwargs,
) -> Array:
  """Batched KL divergence KL(dist1 || dist2) between two Beta distributions.

  Args:
    dist1: A Beta distribution.
    dist2: A Beta distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  alpha1, beta1 = _obtain_alpha_beta(dist1)
  alpha2, beta2 = _obtain_alpha_beta(dist2)
  t1 = math.log_beta(alpha2, beta2) - math.log_beta(alpha1, beta1)
  t2 = (alpha1 - alpha2) * jax.lax.digamma(alpha1)
  t3 = (beta1 - beta2) * jax.lax.digamma(beta1)
  t4 = (alpha2 - alpha1 + beta2 - beta1) * jax.lax.digamma(alpha1 + beta1)
  return t1 + t2 + t3 + t4


# Register the KL functions with TFP.
tfd.RegisterKL(Beta, Beta)(_kl_divergence_beta_beta)
tfd.RegisterKL(Beta, Beta.equiv_tfp_cls)(_kl_divergence_beta_beta)
tfd.RegisterKL(Beta.equiv_tfp_cls, Beta)(_kl_divergence_beta_beta)
