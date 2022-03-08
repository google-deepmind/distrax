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
"""Diagonal affine bijector."""

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import block
from distrax._src.bijectors import scalar_affine
import jax.numpy as jnp

Array = base.Array


def _check_shapes_are_valid(diag: Array, bias: Array):
  """Checks array shapes are valid, raises `ValueError` if not."""
  if diag.ndim < 1:
    raise ValueError("`diag` must have at least one dimension.")
  if bias.ndim < 1:
    raise ValueError("`bias` must have at least one dimension.")
  if bias.shape[-1] != diag.shape[-1]:
    raise ValueError(
        f"Both `bias` and `diag` must have the same number of dimensions; got "
        f"`bias.shape[-1]={bias.shape[-1]}` and "
        f"`diag.shape[-1]={diag.shape[-1]}`.")
  try:
    jnp.broadcast_shapes(diag.shape, bias.shape)
  except ValueError:
    raise ValueError(
        f"The shapes of `bias` and `diag` are not broadcastable; got "
        f"`bias.shape={bias.shape}` and `diag.shape={diag.shape}`.") from None


class DiagAffine(block.Block):
  """Affine bijector with a diagonal weight matrix.

  The bijector is defined as `f(x) = Ax + b` where `A` is a `DxD` diagonal
  matrix and `b` is a `D`-dimensional vector. Additional dimensions, if any,
  index batches.

  The Jacobian determinant is trivially computed by taking the product of the
  diagonal entries in `A`. The inverse transformation `x = f^{-1}(y)` is
  computed element-wise.

  The bijector is invertible if and only if the diagonal entries of `A` are all
  non-zero. It is the responsibility of the user to make sure that this is the
  case; the class will make no attempt to verify that the bijector is
  invertible.
  """

  def __init__(self, diag: Array, bias: Array):
    """Initializes the bijector.

    Args:
      diag: a vector of length D, the diagonal of matrix `A`. Can also be a
        batch of such vectors.
      bias: the vector `b` in `Ax + b`. Can also be a batch of such vectors.
    """
    _check_shapes_are_valid(diag=diag, bias=bias)
    bijector = scalar_affine.ScalarAffine(shift=bias, scale=diag)
    super().__init__(bijector=bijector, ndims=1)
    self._diag = diag
    self._bias = bias

  @property
  def diag(self) -> Array:
    """Vector of length D, the diagonal of matrix `A`."""
    return self._diag

  @property
  def bias(self) -> Array:
    """The bias `b` of the transformation."""
    return self._bias

  @property
  def matrix(self) -> Array:
    """The full matrix `A`."""
    return jnp.vectorize(jnp.diag, signature="(k)->(k,k)")(self.diag)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is DiagAffine:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.diag is other.diag,
          self.bias is other.bias,
      ))
    return False
