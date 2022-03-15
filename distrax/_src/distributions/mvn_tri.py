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
"""MultivariateNormalTri distribution."""

from typing import Optional

import chex
from distrax._src.bijectors.diag_linear import DiagLinear
from distrax._src.bijectors.triangular_linear import TriangularLinear
from distrax._src.distributions import distribution
from distrax._src.distributions.mvn_from_bijector import MultivariateNormalFromBijector
from distrax._src.utils import conversion
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array


def _check_parameters(
    loc: Optional[Array], scale_tri: Optional[Array]) -> None:
  """Checks that the inputs are correct."""

  if loc is None and scale_tri is None:
    raise ValueError(
        'At least one of `loc` and `scale_tri` must be specified.')

  if loc is not None and loc.ndim < 1:
    raise ValueError('The parameter `loc` must have at least one dimension.')

  if scale_tri is not None and scale_tri.ndim < 2:
    raise ValueError(
        f'The parameter `scale_tri` must have at least two dimensions, but '
        f'`scale_tri.shape = {scale_tri.shape}`.')

  if scale_tri is not None and scale_tri.shape[-1] != scale_tri.shape[-2]:
    raise ValueError(
        f'The parameter `scale_tri` must be a (batched) square matrix, but '
        f'`scale_tri.shape = {scale_tri.shape}`.')

  if loc is not None:
    num_dims = loc.shape[-1]
    if scale_tri is not None and scale_tri.shape[-1] != num_dims:
      raise ValueError(
          f'Shapes are not compatible: `loc.shape = {loc.shape}` and '
          f'`scale_tri.shape = {scale_tri.shape}`.')


class MultivariateNormalTri(MultivariateNormalFromBijector):
  """Multivariate normal distribution on `R^k`.

  The `MultivariateNormalTri` distribution is parameterized by a `k`-length
  location (mean) vector `b` and a (lower or upper) triangular scale matrix `S`
  of size `k x k`. The covariance matrix is `C = S @ S.T`.
  """

  equiv_tfp_cls = tfd.MultivariateNormalTriL

  def __init__(self,
               loc: Optional[Array] = None,
               scale_tri: Optional[Array] = None,
               is_lower: bool = True):
    """Initializes a MultivariateNormalTri distribution.

    Args:
      loc: Mean vector of the distribution of shape `k` (can also be a batch of
        such vectors). If not specified, it defaults to zeros.
      scale_tri: The scale matrix `S`. It must be a `k x k` triangular matrix
        (additional dimensions index batches). If `scale_tri` is not triangular,
        the entries above or below the main diagonal will be ignored. The
        parameter `is_lower` specifies if `scale_tri` is lower or upper
        triangular. It is the responsibility of the user to make sure that
        `scale_tri` only contains non-zero elements in its diagonal; this class
        makes no attempt to verify that. If `scale_tri` is not specified, it
        defaults to the identity.
      is_lower: Indicates if `scale_tri` is lower (if True) or upper (if False)
        triangular.
    """
    loc = None if loc is None else conversion.as_float_array(loc)
    scale_tri = None if scale_tri is None else conversion.as_float_array(
        scale_tri)
    _check_parameters(loc, scale_tri)

    if loc is not None:
      num_dims = loc.shape[-1]
    elif scale_tri is not None:
      num_dims = scale_tri.shape[-1]

    dtype = jnp.result_type(*[x for x in [loc, scale_tri] if x is not None])

    if loc is None:
      loc = jnp.zeros((num_dims,), dtype=dtype)

    if scale_tri is None:
      self._scale_tri = jnp.eye(num_dims, dtype=dtype)
      scale = DiagLinear(diag=jnp.ones(loc.shape[-1:], dtype=dtype))
    else:
      tri_fn = jnp.tril if is_lower else jnp.triu
      self._scale_tri = tri_fn(scale_tri)
      scale = TriangularLinear(matrix=self._scale_tri, is_lower=is_lower)
    self._is_lower = is_lower

    super().__init__(loc=loc, scale=scale)

  @property
  def scale_tri(self) -> Array:
    """Triangular scale matrix `S`."""
    return jnp.broadcast_to(
        self._scale_tri,
        self.batch_shape + self.event_shape + self.event_shape)

  @property
  def is_lower(self) -> bool:
    """Whether the `scale_tri` matrix is lower triangular."""
    return self._is_lower

  def __getitem__(self, index) -> 'MultivariateNormalTri':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return MultivariateNormalTri(
        loc=self.loc[index],
        scale_tri=self.scale_tri[index],
        is_lower=self.is_lower)
