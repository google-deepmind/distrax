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
"""Scalar affine bijector."""

from typing import Optional, Tuple, Union

from distrax._src.bijectors import bijector as base
import jax
import jax.numpy as jnp


Array = base.Array
Numeric = Union[Array, float]


class ScalarAffine(base.Bijector):
  """An affine bijector that acts elementwise.

  The bijector is defined as follows:

  - Forward: `y = scale * x + shift`
  - Forward Jacobian determinant: `log|det J(x)| = log|scale|`
  - Inverse: `x = (y - shift) / scale`
  - Inverse Jacobian determinant: `log|det J(y)| = -log|scale|`

  where `scale` and `shift` are the bijector's parameters.
  """

  def __init__(self,
               shift: Numeric,
               scale: Optional[Numeric] = None,
               log_scale: Optional[Numeric] = None):
    """Initializes a ScalarAffine bijector.

    Args:
      shift: the bijector's shift parameter. Can also be batched.
      scale: the bijector's scale parameter. Can also be batched. NOTE: `scale`
        must be non-zero, otherwise the bijector is not invertible. It is the
        user's responsibility to make sure `scale` is non-zero; the class will
        make no attempt to verify this.
      log_scale: the log of the scale parameter. Can also be batched. If
        specified, the bijector's scale is set equal to `exp(log_scale)`. Unlike
        `scale`, `log_scale` is an unconstrained parameter. NOTE: either `scale`
        or `log_scale` can be specified, but not both. If neither is specified,
        the bijector's scale will default to 1.

    Raises:
      ValueError: if both `scale` and `log_scale` are not None.
    """
    super().__init__(event_ndims_in=0, is_constant_jacobian=True)
    self._shift = shift
    if scale is None and log_scale is None:
      self._scale = 1.
      self._inv_scale = 1.
      self._log_scale = 0.
    elif log_scale is None:
      self._scale = scale
      self._inv_scale = 1. / scale
      self._log_scale = jnp.log(jnp.abs(scale))
    elif scale is None:
      self._scale = jnp.exp(log_scale)
      self._inv_scale = jnp.exp(jnp.negative(log_scale))
      self._log_scale = log_scale
    else:
      raise ValueError(
          'Only one of `scale` and `log_scale` can be specified, not both.')
    self._batch_shape = jax.lax.broadcast_shapes(
        jnp.shape(self._shift), jnp.shape(self._scale))

  @property
  def shift(self) -> Numeric:
    """The bijector's shift."""
    return self._shift

  @property
  def log_scale(self) -> Numeric:
    """The log of the bijector's scale."""
    return self._log_scale

  @property
  def scale(self) -> Numeric:
    """The bijector's scale."""
    return self._scale

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    batch_shape = jax.lax.broadcast_shapes(self._batch_shape, x.shape)
    batched_scale = jnp.broadcast_to(self._scale, batch_shape)
    batched_shift = jnp.broadcast_to(self._shift, batch_shape)
    return batched_scale * x + batched_shift

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    batch_shape = jax.lax.broadcast_shapes(self._batch_shape, x.shape)
    return jnp.broadcast_to(self._log_scale, batch_shape)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self.forward(x), self.forward_log_det_jacobian(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    batch_shape = jax.lax.broadcast_shapes(self._batch_shape, y.shape)
    batched_inv_scale = jnp.broadcast_to(self._inv_scale, batch_shape)
    batched_shift = jnp.broadcast_to(self._shift, batch_shape)
    return batched_inv_scale * (y - batched_shift)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    batch_shape = jax.lax.broadcast_shapes(self._batch_shape, y.shape)
    return jnp.broadcast_to(jnp.negative(self._log_scale), batch_shape)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is ScalarAffine:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.shift is other.shift,
          self.scale is other.scale,
          self.log_scale is other.log_scale,
      ))
    else:
      return False
