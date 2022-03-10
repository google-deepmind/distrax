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
"""MultivariateNormalFullCovariance distribution."""

from typing import Optional

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions.mvn_tri import MultivariateNormalTri
from distrax._src.utils import conversion
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
Array = chex.Array


def _check_parameters(
    loc: Optional[Array], covariance_matrix: Optional[Array]) -> None:
  """Checks that the inputs are correct."""

  if loc is None and covariance_matrix is None:
    raise ValueError(
        'At least one of `loc` and `covariance_matrix` must be specified.')

  if loc is not None and loc.ndim < 1:
    raise ValueError('The parameter `loc` must have at least one dimension.')

  if covariance_matrix is not None and covariance_matrix.ndim < 2:
    raise ValueError(
        f'The `covariance_matrix` must have at least two dimensions, but '
        f'`covariance_matrix.shape = {covariance_matrix.shape}`.')

  if covariance_matrix is not None and (
      covariance_matrix.shape[-1] != covariance_matrix.shape[-2]):
    raise ValueError(
        f'The `covariance_matrix` must be a (batched) square matrix, but '
        f'`covariance_matrix.shape = {covariance_matrix.shape}`.')

  if loc is not None:
    num_dims = loc.shape[-1]
    if covariance_matrix is not None and (
        covariance_matrix.shape[-1] != num_dims):
      raise ValueError(
          f'Shapes are not compatible: `loc.shape = {loc.shape}` and '
          f'`covariance_matrix.shape = {covariance_matrix.shape}`.')


class MultivariateNormalFullCovariance(MultivariateNormalTri):
  """Multivariate normal distribution on `R^k`.

  The `MultivariateNormalFullCovariance` distribution is parameterized by a
  `k`-length location (mean) vector `b` and a covariance matrix `C` of size
  `k x k` that must be positive definite and symmetric.

  This class makes no attempt to verify that `C` is positive definite or
  symmetric. It is the responsibility of the user to make sure that it is the
  case.
  """

  equiv_tfp_cls = tfd.MultivariateNormalFullCovariance

  def __init__(self,
               loc: Optional[Array] = None,
               covariance_matrix: Optional[Array] = None):
    """Initializes a MultivariateNormalFullCovariance distribution.

    Args:
      loc: Mean vector of the distribution of shape `k` (can also be a batch of
        such vectors). If not specified, it defaults to zeros.
      covariance_matrix: The covariance matrix `C`. It must be a `k x k` matrix
        (additional dimensions index batches). If not specified, it defaults to
        the identity.
    """
    loc = None if loc is None else conversion.as_float_array(loc)
    covariance_matrix = None if covariance_matrix is None else (
        conversion.as_float_array(covariance_matrix))
    _check_parameters(loc, covariance_matrix)

    if loc is not None:
      num_dims = loc.shape[-1]
    elif covariance_matrix is not None:
      num_dims = covariance_matrix.shape[-1]

    dtype = jnp.result_type(
        *[x for x in [loc, covariance_matrix] if x is not None])

    if loc is None:
      loc = jnp.zeros((num_dims,), dtype=dtype)

    if covariance_matrix is None:
      self._covariance_matrix = jnp.eye(num_dims, dtype=dtype)
      scale_tril = None
    else:
      self._covariance_matrix = covariance_matrix
      scale_tril = jnp.linalg.cholesky(covariance_matrix)

    super().__init__(loc=loc, scale_tri=scale_tril)

  @property
  def covariance_matrix(self) -> Array:
    """Covariance matrix `C`."""
    return jnp.broadcast_to(
        self._covariance_matrix,
        self.batch_shape + self.event_shape + self.event_shape)

  def covariance(self) -> Array:
    """Covariance matrix `C`."""
    return self.covariance_matrix

  def variance(self) -> Array:
    """Calculates the variance of all one-dimensional marginals."""
    return jnp.vectorize(jnp.diag, signature='(k,k)->(k)')(
        self.covariance_matrix)

  def __getitem__(self, index) -> 'MultivariateNormalFullCovariance':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return MultivariateNormalFullCovariance(
        loc=self.loc[index],
        covariance_matrix=self.covariance_matrix[index])
