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
"""LU-decomposed affine bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import unconstrained_affine
import jax
import jax.numpy as jnp


Array = base.Array


class LowerUpperTriangularAffine(base.Bijector):
  """An affine bijector whose weight matrix is parameterized as A = LU.

  This bijector is defined as `f(x) = Ax + b` where:

  * A = LU is a DxD matrix.
  * L is a lower-triangular matrix with ones on the diagonal.
  * U is an upper-triangular matrix.

  The Jacobian determinant can be computed in O(D) as follows:

  log|det J(x)| = log|det A| = sum(log|diag(U)|)

  The inverse can be computed in O(D^2) by solving two triangular systems:

  * Lz = y - b
  * Ux = z

  The bijector is invertible if and only if all diagonal elements of U are
  non-zero. It is the responsibility of the user to make sure that this is the
  case; the class will make no attempt to verify that the bijector is
  invertible.

  L and U are parameterized using a square matrix M as follows:

  * The lower-triangular part of M (excluding the diagonal) becomes L.
  * The upper-triangular part of M (including the diagonal) becomes U.

  The parameterization is such that if M is the identity, LU is also the
  identity. Note however that M is not generally equal to LU.
  """

  def __init__(self, matrix: Array, bias: Array):
    """Initializes a LowerUpperTriangularAffine bijector.

    Args:
      matrix: a square matrix parameterizing `L` and `U` as described in the
        class docstring. Can also be a batch of matrices. If `matrix` is the
        identity, `LU` is also the identity. Note however that `matrix` is
        generally not equal to the product `LU`.
      bias: the vector `b` in `LUx + b`. Can also be a batch of vectors.
    """
    super().__init__(event_ndims_in=1, is_constant_jacobian=True)
    self._batch_shape = unconstrained_affine.common_batch_shape(matrix, bias)
    self._bias = bias

    def compute_lu(matrix):
      # Lower-triangular matrix with ones on the diagonal.
      lower = jnp.eye(matrix.shape[-1]) + jnp.tril(matrix, -1)
      # Upper-triangular matrix.
      upper = jnp.triu(matrix)
      # Log absolute determinant.
      logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(matrix))))
      return lower, upper, logdet

    compute_lu = jnp.vectorize(compute_lu, signature="(m,m)->(m,m),(m,m),()")
    self._lower, self._upper, self._logdet = compute_lu(matrix)

  @property
  def lower(self) -> Array:
    """The lower triangular matrix `L` with ones in the diagonal."""
    return self._lower

  @property
  def upper(self) -> Array:
    """The upper triangular matrix `U`."""
    return self._upper

  @property
  def bias(self) -> Array:
    """The shift `b` of the transformation."""
    return self._bias

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)

    def unbatched(single_x, lower, upper, bias):
      return lower @ (upper @ single_x) + bias

    batched = jnp.vectorize(unbatched, signature="(m),(m,m),(m,m),(m)->(m)")
    return batched(x, self._lower, self._upper, self._bias)

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    batch_shape = jax.lax.broadcast_shapes(self._batch_shape, x.shape[:-1])
    return jnp.broadcast_to(self._logdet, batch_shape)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self.forward(x), self.forward_log_det_jacobian(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    self._check_inverse_input_shape(y)

    def unbatched(single_y, lower, upper, bias):
      x = single_y - bias
      x = jax.scipy.linalg.solve_triangular(
          lower, x, lower=True, unit_diagonal=True)
      x = jax.scipy.linalg.solve_triangular(
          upper, x, lower=False, unit_diagonal=False)
      return x

    batched = jnp.vectorize(unbatched, signature="(m),(m,m),(m,m),(m)->(m)")
    return batched(y, self._lower, self._upper, self._bias)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return -self.forward_log_det_jacobian(y)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)
