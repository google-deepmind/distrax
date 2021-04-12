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
"""MultivariateNormalDiag distribution."""

import math
from typing import Optional, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey

_half_log2pi = 0.5 * math.log(2 * math.pi)


class MultivariateNormalDiag(distribution.Distribution):
  """Multivariate normal distribution on `R^k`."""

  equiv_tfp_cls = tfd.MultivariateNormalDiag

  def __init__(self,
               loc: Optional[Array] = None,
               scale_diag: Optional[Array] = None):
    """Initializes a MultivariateNormalDiag distribution.

    Args:
      loc: Mean vector of the distribution. Can also be a batch of vectors. If
        not specified, it defaults to zeros. At least one of `loc` and
        `scale_diag` must be specified.
      scale_diag: Vector of standard deviations. Can also be a batch of vectors.
        If not specified, it defaults to ones. At least one of `loc` and
        `scale_diag` must be specified.
    """
    super().__init__()
    chex.assert_not_both_none(loc, scale_diag)
    if scale_diag is not None and not scale_diag.shape:
      raise ValueError('If provided, argument `scale_diag` must have at least '
                       '1 dimension.')
    if loc is not None and not loc.shape:
      raise ValueError('If provided, argument `loc` must have at least '
                       '1 dimension.')
    if loc is not None and scale_diag is not None and (
        loc.shape[-1] != scale_diag.shape[-1]):
      raise ValueError(f'The last dimension of arguments `loc` and '
                       f'`scale_diag` must coincide, but {loc.shape[-1]} != '
                       f'{scale_diag.shape[-1]}.')

    if scale_diag is None:
      self._loc = conversion.as_float_array(loc)
      self._scale_diag = jnp.ones(self._loc.shape[-1], self._loc.dtype)
    elif loc is None:
      self._scale_diag = conversion.as_float_array(scale_diag)
      self._loc = jnp.zeros(self._scale_diag.shape[-1], self._scale_diag.dtype)
    else:
      self._loc = conversion.as_float_array(loc)
      self._scale_diag = conversion.as_float_array(scale_diag)

    self._batch_shape = jax.lax.broadcast_shapes(
        self._loc.shape[:-1], self._scale_diag.shape[:-1])

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return (self._num_dims(),)

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def _parameters_shape(self) -> Tuple[int, ...]:
    return self.batch_shape + self.event_shape

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._loc, self._parameters_shape)

  @property
  def scale_diag(self) -> Array:
    """Scale of the distribution."""
    return jnp.broadcast_to(self._scale_diag, self._parameters_shape)

  def _num_dims(self) -> int:
    """Dimensionality of the events."""
    return self._scale_diag.shape[-1]

  def _sample_from_std_normal(self, key: PRNGKey, n: int) -> Array:
    out_shape = (n,) + self._parameters_shape
    dtype = jnp.result_type(self._loc, self._scale_diag)
    return jax.random.normal(key, shape=out_shape, dtype=dtype)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    rnd = self._sample_from_std_normal(key, n)
    return self._loc + self._scale_diag * rnd

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    rnd = self._sample_from_std_normal(key, n)
    samples = self._loc + self._scale_diag * rnd
    log_prob = jnp.sum(
        -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self._scale_diag),
        axis=-1)
    return samples, log_prob

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    log_unnormalized = -0.5 * jnp.square(self._standardize(value))
    log_normalization = _half_log2pi + jnp.log(self._scale_diag)
    return jnp.sum(log_unnormalized - log_normalization, axis=-1)

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jnp.prod(jax.scipy.special.ndtr(self._standardize(value)), axis=-1)

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.sum(
        jax.scipy.special.log_ndtr(self._standardize(value)), axis=-1)

  def _standardize(self, value: Array) -> Array:
    return (value - self._loc) / self._scale_diag

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (nats)."""
    return (
        self._num_dims() * (0.5 + _half_log2pi)
        + jnp.sum(jnp.log(self.scale_diag), axis=-1))

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def median(self) -> Array:
    """Calculates the median."""
    return self.mean()

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self.scale_diag

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self.stddev())

  def covariance(self) -> Array:
    """Calculates the covariance.

    Constructs a diagonal matrix with the variance vector as diagonal. Note that
    TFP would drop leading dimensions in the covariance if
    `self._scale_diag.ndims < self._loc.ndims`. To keep things simple and
    predictable, and for consistency with other distributions, in Distrax the
    `covariance` has shape `batch_shape + (num_dims, num_dims)`.

    Returns:
      Diagonal covariance matrix.
    """
    return jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(self.variance())


def _kl_divergence_mvndiag_mvndiag(
    dist1: Union[MultivariateNormalDiag, tfd.MultivariateNormalDiag],
    dist2: Union[MultivariateNormalDiag, tfd.MultivariateNormalDiag],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Batched divergence KL(dist1 || dist2) between two MultivariateNormalDiag.

  Args:
    dist1: A MultivariateNormalDiag distribution.
    dist2: A MultivariateNormalDiag distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  # pylint: disable=protected-access
  def get_loc_parameter(dist):
    # Converting to jnp array is needed for compatibility with TFP.
    return 0.0 if dist._loc is None else jnp.asarray(dist._loc)

  def get_scale_diag_parameter(dist):
    if isinstance(dist, MultivariateNormalDiag):
      scale_diag = dist._scale_diag
    else:
      # TFP distributions do not have the `_scale_diag` property.
      scale_diag = dist.parameters['scale_diag']
    if scale_diag is None:
      return jnp.ones(shape=dist.event_shape, dtype=dist.dtype)
    return jnp.asarray(scale_diag)

  dist1_loc = get_loc_parameter(dist1)
  dist2_loc = get_loc_parameter(dist2)
  dist1_scale = get_scale_diag_parameter(dist1)
  dist2_scale = get_scale_diag_parameter(dist2)

  diff_log_scale = jnp.log(dist1_scale) - jnp.log(dist2_scale)
  return jnp.sum(
      0.5 * jnp.square(dist1_loc / dist2_scale - dist2_loc / dist2_scale) +
      0.5 * jnp.expm1(2. * diff_log_scale) -
      diff_log_scale, axis=-1)


# Register the KL functions with TFP.
tfd.RegisterKL(MultivariateNormalDiag, MultivariateNormalDiag)(
    _kl_divergence_mvndiag_mvndiag)
tfd.RegisterKL(MultivariateNormalDiag, MultivariateNormalDiag.equiv_tfp_cls)(
    _kl_divergence_mvndiag_mvndiag)
tfd.RegisterKL(MultivariateNormalDiag.equiv_tfp_cls, MultivariateNormalDiag)(
    _kl_divergence_mvndiag_mvndiag)
