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
"""MultivariateNormalDiagPlusLowRank distribution."""

from typing import Optional

import chex
from distrax._src.bijectors.diag_linear import DiagLinear
from distrax._src.bijectors.diag_plus_low_rank_linear import DiagPlusLowRankLinear
from distrax._src.distributions import distribution
from distrax._src.distributions.mvn_from_bijector import MultivariateNormalFromBijector
from distrax._src.utils import conversion
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array


def _check_parameters(
    loc: Optional[Array],
    scale_diag: Optional[Array],
    scale_u_matrix: Optional[Array],
    scale_v_matrix: Optional[Array]) -> None:
  """Checks that the inputs are correct."""

  if all(x is None for x in [loc, scale_diag, scale_u_matrix]):
    raise ValueError(
        'At least one of `loc`, `scale_diag`,  and `scale_u_matrix` must '
        'be specified.')
  if scale_v_matrix is not None and scale_u_matrix is None:
    raise ValueError('`scale_v_matrix` can be specified only when '
                     '`scale_u_matrix` is also specified.')

  for name, x, n_dims in [
      ('loc', loc, 1), ('scale_diag', scale_diag, 1),
      ('scale_u_matrix', scale_u_matrix, 2),
      ('scale_v_matrix', scale_v_matrix, 2)]:
    if x is not None and x.ndim < n_dims:
      raise ValueError(f'`{name}` must have at least {n_dims} dimensions.')

  if scale_u_matrix is not None and scale_v_matrix is not None:
    if scale_u_matrix.shape[-1] != scale_v_matrix.shape[-1]:
      raise ValueError(
          f'The last dimension of `scale_u_matrix` must coincide with '
          f'the last dimension of `scale_v_matrix`, but got '
          f'`scale_u_matrix.shape[-1] = {scale_u_matrix.shape[-1]}`'
          f' and `scale_v_matrix.shape[-1] = {scale_v_matrix.shape[-1]}`.')

  if scale_u_matrix is not None and scale_u_matrix.shape[-1] < 1:
    raise ValueError(
        'The last dimension of `scale_u_matrix` cannot be zero.')

  loc_dim = None if loc is None else loc.shape[-1]
  scale_diag_dim = None if scale_diag is None else scale_diag.shape[-1]
  scale_u_matrix_dim = (
      None if scale_u_matrix is None else scale_u_matrix.shape[-2])
  scale_v_matrix_dim = (
      None if scale_v_matrix is None else scale_v_matrix.shape[-2])
  num_dims = loc_dim if loc_dim is not None else scale_diag_dim
  num_dims = num_dims if num_dims is not None else scale_u_matrix_dim
  array_dims = [
      x for x in [
          loc_dim, scale_diag_dim, scale_u_matrix_dim, scale_v_matrix_dim]
      if x is not None]
  if not all(x == num_dims for x in array_dims):
    raise ValueError(
        f'If specified, the following shapes must all coincide, but got '
        f'`loc.shape[-1] = {loc_dim}`, '
        f'`scale_diag.shape[-1] = {scale_diag_dim}`, '
        f'`scale_u_matrix.shape[-2] = {scale_u_matrix_dim}`, and '
        f'`scale_v_matrix.shape[-2] = {scale_v_matrix_dim}`.')


class MultivariateNormalDiagPlusLowRank(MultivariateNormalFromBijector):
  """Multivariate normal distribution on `R^k`.

  The `MultivariateNormalDiagPlusLowRank` distribution is parameterized by a
  location (mean) vector `b` and a scale matrix `S` that has the following
  structure: `S = diag(D) + U @ V.T`, where `D` is a `k`-length vector, and both
  `U` and `V` are `k x r` matrices (with `r < k` typically). The covariance
  matrix of the multivariate normal distribution is `C = S @ S.T`.

  This class makes no attempt to verify that the scale matrix `S` is invertible,
  which happens if and only if both `diag(D)` and `I + V^T diag(D)^{-1} U` are
  invertible. It is the responsibility of the user to make sure that this is the
  case.
  """

  equiv_tfp_cls = tfd.MultivariateNormalDiagPlusLowRank

  def __init__(self,
               loc: Optional[Array] = None,
               scale_diag: Optional[Array] = None,
               scale_u_matrix: Optional[Array] = None,
               scale_v_matrix: Optional[Array] = None):
    """Initializes a MultivariateNormalDiagPlusLowRank distribution.

    Args:
      loc: Mean vector of the distribution of shape `k` (can also be a batch of
        such vectors). If not specified, it defaults to zeros.
      scale_diag: The diagonal matrix added to the scale `S`, specified by a
        `k`-length vector containing its diagonal entries (or a batch of
        vectors). If not specified, the diagonal matrix defaults to the
        identity.
      scale_u_matrix: The low-rank matrix `U` that specifies the scale, as
        described in the class docstring. It is a `k x r` matrix (or a batch of
        such matrices). If not specified, it defaults to zeros. At least one
        of `loc`, `scale_diag`, and `scale_u_matrix` must be specified.
      scale_v_matrix: The low-rank matrix `V` that specifies the scale, as
        described in the class docstring. It is a `k x r` matrix (or a batch of
        such matrices). If not specified, it defaults to `scale_u_matrix`. It
        can only be specified if `scale_u_matrix` is also specified.
    """
    loc = None if loc is None else conversion.as_float_array(loc)
    scale_diag = None if scale_diag is None else conversion.as_float_array(
        scale_diag)
    scale_u_matrix = (
        None if scale_u_matrix is None else conversion.as_float_array(
            scale_u_matrix))
    scale_v_matrix = (
        None if scale_v_matrix is None else conversion.as_float_array(
            scale_v_matrix))

    _check_parameters(loc, scale_diag, scale_u_matrix, scale_v_matrix)

    if loc is not None:
      num_dims = loc.shape[-1]
    elif scale_diag is not None:
      num_dims = scale_diag.shape[-1]
    elif scale_u_matrix is not None:
      num_dims = scale_u_matrix.shape[-2]

    dtype = jnp.result_type(
        *[x for x in [loc, scale_diag, scale_u_matrix, scale_v_matrix]
          if x is not None])

    if loc is None:
      loc = jnp.zeros((num_dims,), dtype=dtype)

    self._scale_diag = scale_diag
    if scale_diag is None:
      self._scale_diag = jnp.ones((num_dims,), dtype=dtype)

    self._scale_u_matrix = scale_u_matrix
    if scale_u_matrix is None:
      self._scale_u_matrix = jnp.zeros((num_dims, 1), dtype=dtype)

    self._scale_v_matrix = scale_v_matrix
    if scale_v_matrix is None:
      self._scale_v_matrix = self._scale_u_matrix

    if scale_u_matrix is None:
      # The scale matrix is diagonal.
      scale = DiagLinear(self._scale_diag)
    else:
      scale = DiagPlusLowRankLinear(
          u_matrix=self._scale_u_matrix,
          v_matrix=self._scale_v_matrix,
          diag=self._scale_diag)
    super().__init__(loc=loc, scale=scale)

  @property
  def scale_diag(self) -> Array:
    """Diagonal matrix that is added to the scale."""
    return jnp.broadcast_to(
        self._scale_diag, self.batch_shape + self.event_shape)

  @property
  def scale_u_matrix(self) -> Array:
    """Matrix `U` that defines the low-rank part of the scale matrix."""
    return jnp.broadcast_to(
        self._scale_u_matrix,
        self.batch_shape + self._scale_u_matrix.shape[-2:])

  @property
  def scale_v_matrix(self) -> Array:
    """Matrix `V` that defines the low-rank part of the scale matrix."""
    return jnp.broadcast_to(
        self._scale_v_matrix,
        self.batch_shape + self._scale_v_matrix.shape[-2:])

  def __getitem__(self, index) -> 'MultivariateNormalDiagPlusLowRank':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return MultivariateNormalDiagPlusLowRank(
        loc=self.loc[index],
        scale_diag=self.scale_diag[index],
        scale_u_matrix=self.scale_u_matrix[index],
        scale_v_matrix=self.scale_v_matrix[index])
