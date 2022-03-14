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
"""Unconstrained affine bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
import jax
import jax.numpy as jnp

Array = base.Array


def check_affine_parameters(matrix: Array, bias: Array) -> None:
  """Checks that `matrix` and `bias` have valid shapes.

  Args:
    matrix: a matrix, or a batch of matrices.
    bias: a vector, or a batch of vectors.

  Raises:
    ValueError: if the shapes of `matrix` and `bias` are invalid.
  """
  if matrix.ndim < 2:
    raise ValueError(f"`matrix` must have at least 2 dimensions, got"
                     f" {matrix.ndim}.")
  if bias.ndim < 1:
    raise ValueError("`bias` must have at least 1 dimension.")
  if matrix.shape[-2] != matrix.shape[-1]:
    raise ValueError(f"`matrix` must be square; instead, it has shape"
                     f" {matrix.shape[-2:]}.")
  if matrix.shape[-1] != bias.shape[-1]:
    raise ValueError(f"`matrix` and `bias` have inconsistent shapes: `matrix`"
                     f" is {matrix.shape[-2:]}, `bias` is {bias.shape[-1:]}.")


class UnconstrainedAffine(base.Bijector):
  """An unconstrained affine bijection.

  This bijector is a linear-plus-bias transformation `f(x) = Ax + b`, where `A`
  is a `D x D` square matrix and `b` is a `D`-dimensional vector.

  The bijector is invertible if and only if `A` is an invertible matrix. It is
  the responsibility of the user to make sure that this is the case; the class
  will make no attempt to verify that the bijector is invertible.

  The Jacobian determinant is equal to `det(A)`. The inverse is computed by
  solving the linear system `Ax = y - b`.

  WARNING: Both the determinant and the inverse cost `O(D^3)` to compute. Thus,
  this bijector is recommended only for small `D`.
  """

  def __init__(self, matrix: Array, bias: Array):
    """Initializes an `UnconstrainedAffine` bijector.

    Args:
      matrix: the matrix `A` in `Ax + b`. Must be square and invertible. Can
        also be a batch of matrices.
      bias: the vector `b` in `Ax + b`. Can also be a batch of vectors.
    """
    check_affine_parameters(matrix, bias)
    super().__init__(event_ndims_in=1, is_constant_jacobian=True)
    self._batch_shape = jnp.broadcast_shapes(matrix.shape[:-2], bias.shape[:-1])
    self._matrix = matrix
    self._bias = bias
    self._logdet = jnp.linalg.slogdet(matrix)[1]

  @property
  def matrix(self) -> Array:
    """The matrix `A` of the transformation."""
    return self._matrix

  @property
  def bias(self) -> Array:
    """The shift `b` of the transformation."""
    return self._bias

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)

    def unbatched(single_x, matrix, bias):
      return matrix @ single_x + bias

    batched = jnp.vectorize(unbatched, signature="(m),(m,m),(m)->(m)")
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

    def unbatched(single_y, matrix, bias):
      return jnp.linalg.solve(matrix, single_y - bias)

    batched = jnp.vectorize(unbatched, signature="(m),(m,m),(m)->(m)")
    return batched(y, self._matrix, self._bias)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return -self.forward_log_det_jacobian(y)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is UnconstrainedAffine:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.matrix is other.matrix,
          self.bias is other.bias,
      ))
    return False
