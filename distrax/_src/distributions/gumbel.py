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
"""Gumbel distribution."""

import math
from typing import Tuple, Union

import chex
from distrax._src.distributions import transformed, uniform
from distrax._src.bijectors import gumbel, inverse
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Gumbel(transformed.Transformed):
  """Gumbel distribution with location `loc` and `scale` parameters."""

  equiv_tfp_cls = tfd.Gumbel

  def __init__(self, loc: Numeric, scale: Numeric):
    """Initializes a Gumbel distribution.

    Args:
      loc: Mean of the distribution.
      scale: Spread of the distribution.
    """
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._loc.shape, self._scale.shape)
    gumbel_bijector = gumbel.GumbelCDF(loc=self._loc, scale=self._scale)

    dtype = jnp.result_type(self._loc, self._scale)
    super().__init__(
        distribution=uniform.Uniform(
            low=jnp.broadcast_to(jnp.finfo(dtype).tiny, self.batch_shape), 
            high=jnp.ones_like(self.loc)),
        bijector=inverse.Inverse(gumbel_bijector))

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
    """Scale of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    z = (value - self._loc) / self._scale
    return  -(z + jnp.exp(-z)) - jnp.log(self._scale)

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    return jnp.log(self._scale) + 1 + jnp.euler_gamma

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jnp.exp(-jnp.exp(-(value - self._loc) / self._scale))

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return -jnp.exp(-(value - self._loc) / self._scale)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._loc + self._scale * jnp.euler_gamma

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self._scale * jnp.ones_like(self._loc) * jnp.pi / jnp.sqrt(6)

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self._scale * jnp.ones_like(self._loc) * jnp.pi) / 6

  def mode(self) -> Array:
    """Calculates the mode."""
    return self._loc * jnp.ones_like(self._scale)

  def median(self) -> Array:
    """Calculates the median."""
    return self._loc - self._scale * jnp.log(jnp.log(2))


def _kl_divergence_gumbel_gumbel(
    dist1: Union[Gumbel, tfd.Gumbel],
    dist2: Union[Gumbel, tfd.Gumbel],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Batched KL divergence KL(dist1 || dist2) between two Gumbel distributions.

  Args:
    dist1: A Gumbel distribution.
    dist2: A Gumbel distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  return (jnp.log(dist2.scale) - jnp.log(dist1.scale) + jnp.euler_gamma *
            (dist1.scale / dist2.scale - 1.) +
            jnp.expm1((dist2.loc - dist1.loc) / dist2.scale +
                          jnp.lgamma(dist1.scale / dist2.scale + 1.)) +
            (dist1.loc - dist2.loc) / dist2.scale)


# Register the KL functions with TFP.
tfd.RegisterKL(Gumbel, Gumbel)(_kl_divergence_gumbel_gumbel)
tfd.RegisterKL(Gumbel, Gumbel.equiv_tfp_cls)(_kl_divergence_gumbel_gumbel)
tfd.RegisterKL(Gumbel.equiv_tfp_cls, Gumbel)(_kl_divergence_gumbel_gumbel)
