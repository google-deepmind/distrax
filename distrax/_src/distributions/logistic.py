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
"""Logistic distribution."""

from typing import Tuple

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


class Logistic(distribution.Distribution):
  """The Logistic distribution with location `loc` and `scale` parameters."""

  equiv_tfp_cls = tfd.Logistic

  def __init__(self, loc: Numeric, scale: Numeric) -> None:
    """Initializes a Logistic distribution.

    Args:
      loc: Mean of the distribution.
      scale: Spread of the distribution.
    """
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._loc.shape, self._scale.shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Spread of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

  def _standardize(self, x: Array) -> Array:
    return (x - self.loc) / self.scale

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._loc, self._scale)
    uniform = jax.random.uniform(
        key,
        shape=out_shape,
        dtype=dtype,
        minval=jnp.finfo(dtype).tiny,
        maxval=1.)
    rnd = jnp.log(uniform) - jnp.log1p(-uniform)
    return self._scale * rnd + self._loc

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    z = self._standardize(value)
    return -z - 2. * jax.nn.softplus(-z) - jnp.log(self._scale)

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in Nats)."""
    return 2. + jnp.broadcast_to(jnp.log(self._scale), self.batch_shape)

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jax.nn.sigmoid(self._standardize(value))

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return -jax.nn.softplus(-self._standardize(value))

  def survival_function(self, value: Array) -> Array:
    """See `Distribution.survival_function`."""
    return jax.nn.sigmoid(-self._standardize(value))

  def log_survival_function(self, value: Array) -> Array:
    """See `Distribution.log_survival_function`."""
    return -jax.nn.softplus(self._standardize(value))

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self.scale * jnp.pi) / 3.

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self.scale * jnp.pi / jnp.sqrt(3.)

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def median(self) -> Array:
    """Calculates the median."""
    return self.mean()

  def __getitem__(self, index) -> 'Logistic':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Logistic(loc=self.loc[index], scale=self.scale[index])
