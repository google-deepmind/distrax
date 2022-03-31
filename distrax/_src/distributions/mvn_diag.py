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

from typing import Optional

import chex
from distrax._src.bijectors.diag_linear import DiagLinear
from distrax._src.distributions import distribution
from distrax._src.distributions.mvn_from_bijector import MultivariateNormalFromBijector
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array


def _check_parameters(
    loc: Optional[Array], scale_diag: Optional[Array]) -> None:
  """Checks that the `loc` and `scale_diag` parameters are correct."""
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


class MultivariateNormalDiag(MultivariateNormalFromBijector):
  """Multivariate normal distribution on `R^k` with diagonal covariance."""

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
    _check_parameters(loc, scale_diag)

    if scale_diag is None:
      loc = conversion.as_float_array(loc)
      scale_diag = jnp.ones(loc.shape[-1], loc.dtype)
    elif loc is None:
      scale_diag = conversion.as_float_array(scale_diag)
      loc = jnp.zeros(scale_diag.shape[-1], scale_diag.dtype)
    else:
      loc = conversion.as_float_array(loc)
      scale_diag = conversion.as_float_array(scale_diag)

    # Add leading dimensions to the paramteters to match the batch shape. This
    # prevents automatic rank promotion.
    broadcasted_shapes = jnp.broadcast_shapes(loc.shape, scale_diag.shape)
    loc = jnp.expand_dims(
        loc, axis=list(range(len(broadcasted_shapes) - loc.ndim)))
    scale_diag = jnp.expand_dims(
        scale_diag, axis=list(range(len(broadcasted_shapes) - scale_diag.ndim)))

    bias = jnp.zeros_like(loc, shape=loc.shape[-1:])
    bias = jnp.expand_dims(
        bias, axis=list(range(len(broadcasted_shapes) - bias.ndim)))
    scale = DiagLinear(scale_diag)
    super().__init__(loc=loc, scale=scale)
    self._scale_diag = scale_diag

  @property
  def scale_diag(self) -> Array:
    """Scale of the distribution."""
    return jnp.broadcast_to(
        self._scale_diag, self.batch_shape + self.event_shape)

  def _standardize(self, value: Array) -> Array:
    return (value - self._loc) / self._scale_diag

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    return jnp.prod(jax.scipy.special.ndtr(self._standardize(value)), axis=-1)

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.sum(
        jax.scipy.special.log_ndtr(self._standardize(value)), axis=-1)

  def __getitem__(self, index) -> 'MultivariateNormalDiag':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return MultivariateNormalDiag(
        loc=self.loc[index], scale_diag=self.scale_diag[index])
