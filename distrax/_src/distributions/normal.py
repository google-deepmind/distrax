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
"""Normal distribution."""

import math
from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey

_half_log2pi = 0.5 * math.log(2 * math.pi)


class Normal(distribution.Distribution):
  """Normal distribution with location `loc` and `scale` parameters."""

  equiv_tfp_cls = tfd.Normal

  def __init__(self, loc: Numeric, scale: Numeric):
    """Initializes a Normal distribution.

    Args:
      loc: Mean of the distribution.
      scale: Standard deviation of the distribution.
    """
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return jax.lax.broadcast_shapes(self._loc.shape, self._scale.shape)

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Scale of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

  def _sample_from_std_normal(self, key: PRNGKey, n: int) -> Array:
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._loc, self._scale)
    return jax.random.normal(key, shape=out_shape, dtype=dtype)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    rnd = self._sample_from_std_normal(key, n)
    scale = jnp.expand_dims(self._scale, range(rnd.ndim - self._scale.ndim))
    loc = jnp.expand_dims(self._loc, range(rnd.ndim - self._loc.ndim))
    return scale * rnd + loc

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """See `Distribution._sample_n_and_log_prob`."""
    rnd = self._sample_from_std_normal(key, n)
    samples = self._scale * rnd + self._loc
    log_prob = -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self._scale)
    return samples, log_prob

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    log_unnormalized = -0.5 * jnp.square(self._standardize(value))
    log_normalization = _half_log2pi + jnp.log(self._scale)
    return log_unnormalized - log_normalization

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jax.scipy.special.ndtr(self._standardize(value))

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jax.scipy.special.log_ndtr(self._standardize(value))

  def survival_function(self, value: Array) -> Array:
    """See `Distribution.survival_function`."""
    return jax.scipy.special.ndtr(-self._standardize(value))

  def log_survival_function(self, value: Array) -> Array:
    """See `Distribution.log_survival_function`."""
    return jax.scipy.special.log_ndtr(-self._standardize(value))

  def _standardize(self, value: Array) -> Array:
    return (value - self._loc) / self._scale

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    log_normalization = _half_log2pi + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self.scale)

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self.scale

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def median(self) -> Array:
    """Calculates the median."""
    return self.mean()

  def __getitem__(self, index) -> 'Normal':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Normal(loc=self.loc[index], scale=self.scale[index])


def _kl_divergence_normal_normal(
    dist1: Union[Normal, tfd.Normal],
    dist2: Union[Normal, tfd.Normal],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Obtain the batched KL divergence KL(dist1 || dist2) between two Normals.

  Args:
    dist1: A Normal distribution.
    dist2: A Normal distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  diff_log_scale = jnp.log(dist1.scale) - jnp.log(dist2.scale)
  return (
      0.5 * jnp.square(dist1.loc / dist2.scale - dist2.loc / dist2.scale) +
      0.5 * jnp.expm1(2. * diff_log_scale) -
      diff_log_scale)


# Register the KL functions with TFP.
tfd.RegisterKL(Normal, Normal)(_kl_divergence_normal_normal)
tfd.RegisterKL(Normal, Normal.equiv_tfp_cls)(_kl_divergence_normal_normal)
tfd.RegisterKL(Normal.equiv_tfp_cls, Normal)(_kl_divergence_normal_normal)
