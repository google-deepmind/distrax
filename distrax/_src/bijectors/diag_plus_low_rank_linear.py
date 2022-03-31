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
"""Diagonal-plus-low-rank linear bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import chain
from distrax._src.bijectors import diag_linear
from distrax._src.bijectors import linear
import jax
import jax.numpy as jnp

Array = base.Array


def _get_small_matrix(u_matrix: Array, v_matrix: Array) -> Array:
  rank = u_matrix.shape[-1]
  return jnp.eye(rank) + v_matrix.T @ u_matrix


def _get_logdet(matrix: Array) -> Array:
  """Computes the log absolute determinant of `matrix`."""
  return jnp.linalg.slogdet(matrix)[1]


def _forward_unbatched(x: Array, u_matrix: Array, v_matrix: Array) -> Array:
  return x + u_matrix @ (v_matrix.T @ x)


def _inverse_unbatched(
    y: Array, u_matrix: Array, v_matrix: Array, small_matrix: Array) -> Array:
  return y - u_matrix @ jax.scipy.linalg.solve(small_matrix, v_matrix.T @ y)


class _IdentityPlusLowRankLinear(base.Bijector):
  """Linear bijector whose weights are a low-rank perturbation of the identity.

  The bijector is defined as `f(x) = Ax` where `A = I + UV^T` and `U`, `V` are
  DxK matrices. When K < D, this bijector is computationally more efficient than
  an equivalent `UnconstrainedAffine` bijector.

  The Jacobian determinant is computed using the matrix determinant lemma:

  det J(x) = det A = det(I + V^T U)

  The matrix `I + V^T U` is KxK instead of DxD, so for K < D computing its
  determinant is faster than computing the determinant of `A`.

  The inverse is computed using the Woodbury matrix identity:

  A^{-1} = I - U (I + V^T U)^{-1} V^T

  As above, inverting the KxK matrix `I + V^T U` is faster than inverting `A`
  when K < D.

  The bijector is invertible if and only if `I + V^T U` is invertible. It is the
  responsibility of the user to make sure that this is the case; the class will
  make no attempt to verify that the bijector is invertible.
  """

  def __init__(self, u_matrix: Array, v_matrix: Array):
    """Initializes the bijector.

    Args:
      u_matrix: a DxK matrix, the `U` matrix in `A = I + UV^T`. Can also be a
        batch of DxK matrices.
      v_matrix: a DxK matrix, the `V` matrix in `A = I + UV^T`. Can also be a
        batch of DxK matrices.
    """
    super().__init__(event_ndims_in=1, is_constant_jacobian=True)
    self._batch_shape = jax.lax.broadcast_shapes(
        u_matrix.shape[:-2], v_matrix.shape[:-2])
    self._u_matrix = u_matrix
    self._v_matrix = v_matrix
    self._small_matrix = jnp.vectorize(
        _get_small_matrix, signature="(d,k),(d,k)->(k,k)")(u_matrix, v_matrix)
    self._logdet = _get_logdet(self._small_matrix)

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)
    batched = jnp.vectorize(
        _forward_unbatched, signature="(d),(d,k),(d,k)->(d)")
    return batched(x, self._u_matrix, self._v_matrix)

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
        _inverse_unbatched, signature="(d),(d,k),(d,k),(k,k)->(d)")
    return batched(y, self._u_matrix, self._v_matrix, self._small_matrix)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return -self.forward_log_det_jacobian(y)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)


def _check_shapes_are_valid(diag: Array,
                            u_matrix: Array,
                            v_matrix: Array) -> None:
  """Checks array shapes are valid, raises `ValueError` if not."""
  for x, name, n in [(diag, "diag", 1),
                     (u_matrix, "u_matrix", 2),
                     (v_matrix, "v_matrix", 2)]:
    if x.ndim < n:
      raise ValueError(
          f"`{name}` must have at least {n} dimensions, got {x.ndim}.")
  dim = diag.shape[-1]
  u_shape = u_matrix.shape[-2:]
  v_shape = v_matrix.shape[-2:]
  if u_shape[0] != dim:
    raise ValueError(
        f"The length of `diag` must equal the first dimension of `u_matrix`. "
        f"Got `diag.length = {dim}` and `u_matrix.shape = {u_shape}`.")
  if u_shape != v_shape:
    raise ValueError(
        f"`u_matrix` and `v_matrix` must have the same shape; got "
        f"`u_matrix.shape = {u_shape}` and `v_matrix.shape = {v_shape}`.")


class DiagPlusLowRankLinear(linear.Linear):
  """Linear bijector whose weights are a low-rank perturbation of a diagonal.

  The bijector is defined as `f(x) = Ax` where `A = S + UV^T` and:
  - `S` is a DxD diagonal matrix,
  - `U`, `V` are DxK matrices.
  When K < D, this bijector is computationally more efficient than an equivalent
  `UnconstrainedAffine` bijector.

  The Jacobian determinant is computed using the matrix determinant lemma:

  det J(x) = det A = det(S) det(I + V^T S^{-1} U)

  The matrix `I + V^T S^{-1} U` is KxK instead of DxD, so for K < D computing
  its determinant is faster than computing the determinant of `A`.

  The inverse is computed using the Woodbury matrix identity:

  A^{-1} = (I - S^{-1} U (I + V^T S^{-1} U)^{-1} V^T) S^{-1}

  As above, inverting the KxK matrix `I + V^T S^{-1} U` is faster than inverting
  `A` when K < D.

  The bijector is invertible if and only if both `S` and `I + V^T S^{-1} U` are
  invertible matrices. It is the responsibility of the user to make sure that
  this is the case; the class will make no attempt to verify that the bijector
  is invertible.
  """

  def __init__(self, diag: Array, u_matrix: Array, v_matrix: Array):
    """Initializes the bijector.

    Args:
      diag: a vector of length D, the diagonal of matrix `S`. Can also be a
        batch of such vectors.
      u_matrix: a DxK matrix, the `U` matrix in `A = S + UV^T`. Can also be a
        batch of DxK matrices.
      v_matrix: a DxK matrix, the `V` matrix in `A = S + UV^T`. Can also be a
        batch of DxK matrices.
    """
    _check_shapes_are_valid(diag, u_matrix, v_matrix)
    # Since `S + UV^T = S (I + WV^T)` where `W = S^{-1}U`, we can implement this
    # bijector by composing `_IdentityPlusLowRankLinear` with `DiagLinear`.
    id_plus_low_rank_linear = _IdentityPlusLowRankLinear(
        u_matrix=u_matrix / diag[..., None],
        v_matrix=v_matrix)
    self._bijector = chain.Chain(
        [diag_linear.DiagLinear(diag), id_plus_low_rank_linear])
    batch_shape = jnp.broadcast_shapes(
        diag.shape[:-1], u_matrix.shape[:-2], v_matrix.shape[:-2])
    dtype = jnp.result_type(diag, u_matrix, v_matrix)
    super().__init__(
        event_dims=diag.shape[-1], batch_shape=batch_shape, dtype=dtype)
    self._diag = diag
    self._u_matrix = u_matrix
    self._v_matrix = v_matrix
    self.forward = self._bijector.forward
    self.forward_log_det_jacobian = self._bijector.forward_log_det_jacobian
    self.inverse = self._bijector.inverse
    self.inverse_log_det_jacobian = self._bijector.inverse_log_det_jacobian
    self.inverse_and_log_det = self._bijector.inverse_and_log_det

  @property
  def diag(self) -> Array:
    """Vector of length D, the diagonal of matrix `S`."""
    return self._diag

  @property
  def u_matrix(self) -> Array:
    """The `U` matrix in `A = S + UV^T`."""
    return self._u_matrix

  @property
  def v_matrix(self) -> Array:
    """The `V` matrix in `A = S + UV^T`."""
    return self._v_matrix

  @property
  def matrix(self) -> Array:
    """The matrix `A = S + UV^T` of the transformation."""
    batched = jnp.vectorize(
        lambda s, u, v: jnp.diag(s) + u @ v.T,
        signature="(d),(d,k),(d,k)->(d,d)")
    return batched(self._diag, self._u_matrix, self._v_matrix)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self._bijector.forward_and_log_det(x)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is DiagPlusLowRankLinear:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.diag is other.diag,
          self.u_matrix is other.u_matrix,
          self.v_matrix is other.v_matrix,
      ))
    return False
