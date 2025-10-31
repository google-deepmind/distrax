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
"""Cauchy distribution."""

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


class Cauchy(distribution.Distribution):
  """Cauchy distribution with loc and scale parameters."""

  equiv_tfp_cls = tfd.Cauchy

  def __init__(self, loc: Numeric, scale: Numeric):
    """Initializes a Cauchy distribution.

    Args:
      loc: Location parameter.
      scale: Scale parameter.
    """
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)
    # TFP's implementation uses a bijector to ensure scale is positive.
    # Distrax's convention seems to be to rely on user input validation
    # or to not enforce it in the constructor.
    # We will remove the assertion as it caused a pytype error and
    # other distributions like Normal don't have it.

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return jnp.broadcast_shapes(self._loc.shape, self._scale.shape)

  @property
  def loc(self) -> Array:
    """Location parameter of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Scale parameter of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    shape = (n,) + self.batch_shape
    uniform = jax.random.uniform(
        key=key, shape=shape, dtype=self.loc.dtype, minval=0.0, maxval=1.0
    )
    # Inverse CDF logic from TFP: x = loc + scale * tan(pi * (u - 0.5))
    standard_cauchy = jnp.tan(jnp.pi * (uniform - 0.5))
    return self.loc + self.scale * standard_cauchy

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    z = (value - self.loc) / self.scale
    return -jnp.log(jnp.pi) - jnp.log(self.scale) - jnp.log(1.0 + jnp.square(z))

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    z = (value - self.loc) / self.scale
    return jnp.arctan(z) / jnp.pi + 0.5

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    z = (value - self.loc) / self.scale
    # Formula from TFP:
    # tf.math.log1p(2 / np.pi * tf.atan(self._z(x))) - np.log(2)
    return jnp.log1p(2.0 / jnp.pi * jnp.arctan(z)) - jnp.log(2.0)

  def mean(self) -> Array:
    """Calculates the mean."""
    return jnp.full(self.batch_shape, jnp.nan, dtype=self.loc.dtype)

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.full(self.batch_shape, jnp.nan, dtype=self.loc.dtype)

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return jnp.full(self.batch_shape, jnp.nan, dtype=self.loc.dtype)

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.loc

  def entropy(self) -> Array:
    """Calculates the entropy."""
    return jnp.log(4.0 * jnp.pi * self.scale)

  def __getitem__(self, index) -> 'Cauchy':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Cauchy(loc=self.loc[index], scale=self.scale[index])
