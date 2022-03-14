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

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import block
from distrax._src.bijectors import chain
from distrax._src.bijectors import shift
from distrax._src.bijectors import triangular_linear
from distrax._src.bijectors import unconstrained_affine
import jax.numpy as jnp

Array = base.Array


class LowerUpperTriangularAffine(chain.Chain):
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
    """Initializes a `LowerUpperTriangularAffine` bijector.

    Args:
      matrix: a square matrix parameterizing `L` and `U` as described in the
        class docstring. Can also be a batch of matrices. If `matrix` is the
        identity, `LU` is also the identity. Note however that `matrix` is
        generally not equal to the product `LU`.
      bias: the vector `b` in `LUx + b`. Can also be a batch of vectors.
    """
    unconstrained_affine.check_affine_parameters(matrix, bias)
    self._upper = triangular_linear.TriangularLinear(matrix, is_lower=False)
    dim = matrix.shape[-1]
    lower = jnp.eye(dim) + jnp.tril(matrix, -1)  # Replace diagonal with ones.
    self._lower = triangular_linear.TriangularLinear(lower, is_lower=True)
    self._shift = block.Block(shift.Shift(bias), 1)
    self._bias = bias
    super().__init__([self._shift, self._lower, self._upper])

  @property
  def lower(self) -> Array:
    """The lower triangular matrix `L` with ones in the diagonal."""
    return self._lower.matrix

  @property
  def upper(self) -> Array:
    """The upper triangular matrix `U`."""
    return self._upper.matrix

  @property
  def matrix(self) -> Array:
    """The matrix `A = LU` of the transformation."""
    return self.lower @ self.upper

  @property
  def bias(self) -> Array:
    """The shift `b` of the transformation."""
    return self._bias

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is LowerUpperTriangularAffine:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.lower is other.lower,
          self.upper is other.upper,
          self.bias is other.bias,
      ))
    return False
