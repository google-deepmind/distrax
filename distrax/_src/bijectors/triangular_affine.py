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
"""Triangular affine bijector."""

import functools
from typing import Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import unconstrained_affine
import jax
import jax.numpy as jnp

Array = base.Array


def _triangular_logdet(matrix: Array) -> Array:
  """Computes the log absolute determinant of a triangular matrix."""
  return jnp.sum(jnp.log(jnp.abs(jnp.diag(matrix))))


def _forward_unbatched(x: Array, matrix: Array, bias: Array) -> Array:
  return matrix @ x + bias


def _inverse_unbatched(
    y: Array, matrix: Array, bias: Array, is_lower: bool) -> Array:
  return jax.scipy.linalg.solve_triangular(matrix, y - bias, lower=is_lower)


class TriangularAffine(base.Bijector):
  """An affine bijector whose weight matrix is triangular.

  The bijector is defined as `f(x) = Ax + b` where `A` is a DxD triangular
  matrix.

  The Jacobian determinant can be computed in O(D) as follows:

  log|det J(x)| = log|det A| = sum(log|diag(A)|)

  The inverse is computed in O(D^2) by solving the triangular system Ax = y - b.

  The bijector is invertible if and only if all diagonal elements of A are
  non-zero. It is the responsibility of the user to make sure that this is the
  case; the class will make no attempt to verify that the bijector is
  invertible.
  """

  def __init__(self, matrix: Array, bias: Array, is_lower: bool = True):
    """Initializes a TriangularAffine bijector.

    Args:
      matrix: a square matrix whose triangular part defines `A` in `Ax + b`. Can
        also be a batch of matrices. Whether `A` is the lower or upper
        triangular part of `matrix` is determined by `is_lower`.
      bias: the vector `b` in `Ax + b`. Can also be a batch of vectors.
      is_lower: if True, `A` is set to the lower triangular part of `matrix`. If
        False, `A` is set to the upper triangular part of `matrix`.
    """
    super().__init__(event_ndims_in=1, is_constant_jacobian=True)
    self._batch_shape = unconstrained_affine.common_batch_shape(matrix, bias)
    self._matrix = jnp.tril(matrix) if is_lower else jnp.triu(matrix)
    self._bias = bias
    self._is_lower = is_lower
    triangular_logdet = jnp.vectorize(_triangular_logdet, signature="(m,m)->()")
    self._logdet = triangular_logdet(self._matrix)

  @property
  def matrix(self) -> Array:
    """The triangular matrix `A` of the transformation."""
    return self._matrix

  @property
  def bias(self) -> Array:
    """The shift `b` of the transformation."""
    return self._bias

  @property
  def is_lower(self) -> bool:
    """True if `A` is lower triangular, False if upper triangular."""
    return self._is_lower

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)
    batched = jnp.vectorize(_forward_unbatched, signature="(m),(m,m),(m)->(m)")
    return batched(x, self._matrix, self._bias)

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
    batched = jnp.vectorize(
        functools.partial(_inverse_unbatched, is_lower=self._is_lower),
        signature="(m),(m,m),(m)->(m)")
    return batched(y, self._matrix, self._bias)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return -self.forward_log_det_jacobian(y)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is TriangularAffine:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.matrix is other.matrix,
          self.bias is other.bias,
          self.is_lower is other.is_lower,
      ))
    return False
