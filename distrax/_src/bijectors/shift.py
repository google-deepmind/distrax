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
"""Shift bijector."""

from typing import Tuple, Union

from distrax._src.bijectors import bijector as base
import jax
import jax.numpy as jnp

Array = base.Array
Numeric = Union[Array, float]


class Shift(base.Bijector):
  """Bijector that translates its input elementwise.

  The bijector is defined as follows:

  - Forward: `y = x + shift`
  - Forward Jacobian determinant: `log|det J(x)| = 0`
  - Inverse: `x = y - shift`
  - Inverse Jacobian determinant: `log|det J(y)| = 0`

  where `shift` parameterizes the bijector.
  """

  def __init__(self, shift: Numeric):
    """Initializes a `Shift` bijector.

    Args:
      shift: the bijector's shift parameter. Can also be batched.
    """
    super().__init__(event_ndims_in=0, is_constant_jacobian=True)
    self._shift = shift
    self._batch_shape = jnp.shape(self._shift)

  @property
  def shift(self) -> Numeric:
    """The bijector's shift."""
    return self._shift

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    return x + self._shift

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    batch_shape = jax.lax.broadcast_shapes(self._batch_shape, x.shape)
    return jnp.zeros(batch_shape, dtype=x.dtype)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self.forward(x), self.forward_log_det_jacobian(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return y - self._shift

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return self.forward_log_det_jacobian(y)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Shift:  # pylint: disable=unidiomatic-typecheck
      return self.shift is other.shift
    return False
