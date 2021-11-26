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
"""Distrax bijector for automatically turning JAX functions into Bijectors."""

from typing import Callable, Optional, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import transformations

Array = base.Array


class Lambda(base.Bijector):
  """Wrapper to automatically turn JAX functions into fully fledged bijectors.

  This class takes in JAX functions that implement bijector methods (such as
  `forward`, `inverse`, `forward_log_det_jacobian`, etc.), and constructs a
  bijector out of them. Any functions not explicitly specified by the user will
  be automatically derived from the existing functions where possible, by
  tracing their JAXPR representation. Missing functions will be derived on
  demand: if a missing function is not used, it will not be derived. At a
  minimum, either `forward` or `inverse` must be given; all other methods will
  be derived (where possible).

  The Lambda bijector can be useful for creating simple one-line bijectors that
  would otherwise be tedious to define. Examples of scalar bijectors that can be
  easily constructed with Lambda are:

  - Identity: `Lambda(lambda x: x)`
  - Affine: `Lambda(lambda x: a*x + b)`
  - Tanh: `Lambda(jnp.tanh)`
  - Composite: `Lambda(lambda x: jnp.tanh(a*x + b))`

  Requirements and limitations:

  - Only functions composed entirely of invertible primitives can be
    automatically inverted (see `bijection_utils.py` for a list of invertible
    primitives). If the inverse is needed but is not automatically derivable,
    the user must provide it explicitly.

  - If log-determinant functions are not provided, Lambda will assume that
    `forward` and `inverse` are scalar functions applied elementwise. If the
    bijector is not meant to be scalar, its log-determinant functions must be
    provided explicitly by the user.
  """

  def __init__(
      self,
      forward: Optional[Callable[[Array], Array]] = None,
      inverse: Optional[Callable[[Array], Array]] = None,
      forward_log_det_jacobian: Optional[Callable[[Array], Array]] = None,
      inverse_log_det_jacobian: Optional[Callable[[Array], Array]] = None,
      event_ndims_in: Optional[int] = None,
      event_ndims_out: Optional[int] = None,
      is_constant_jacobian: Optional[bool] = None):
    """Initializes a Lambda bijector with methods specified as args."""

    if forward is None and inverse is None:
      raise ValueError("The Lambda bijector requires at least one of `forward` "
                       "or `inverse` to be specified, but neither is.")

    jac_functions_specified = (forward_log_det_jacobian is not None
                               or inverse_log_det_jacobian is not None)
    if jac_functions_specified:
      if event_ndims_in is None:
        raise ValueError("When log det Jacobian functions are specified, you "
                         "must also specify `event_ndims_in`.")
    else:
      if event_ndims_in is not None or event_ndims_out is not None:
        raise ValueError("When log det Jacobian functions are unspecified, you "
                         "must leave `event_ndims_in` and `event_ndims_out` "
                         "unspecified; they will default to 0.")
      event_ndims_in = 0

    if is_constant_jacobian is None:
      fn = inverse if forward is None else forward
      is_constant_jacobian = transformations.is_constant_jacobian(fn)

    super().__init__(
        event_ndims_in=event_ndims_in,
        event_ndims_out=event_ndims_out,
        is_constant_jacobian=is_constant_jacobian)
    self._forward = forward
    self._inverse = inverse
    self._forward_log_det_jacobian = forward_log_det_jacobian
    self._inverse_log_det_jacobian = inverse_log_det_jacobian

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)
    if self._forward is None:
      self._forward = transformations.inv(self._inverse)
    return self._forward(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    self._check_inverse_input_shape(y)
    if self._inverse is None:
      self._inverse = transformations.inv(self._forward)
    return self._inverse(y)

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    if self._forward_log_det_jacobian is None:
      self._forward_log_det_jacobian = transformations.log_det_scalar(
          self.forward)
    return self._forward_log_det_jacobian(x)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    if self._inverse_log_det_jacobian is None:
      self._inverse_log_det_jacobian = transformations.log_det_scalar(
          self.inverse)
    return self._inverse_log_det_jacobian(y)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self.forward(x), self.forward_log_det_jacobian(x)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self.inverse(y), self.inverse_log_det_jacobian(y)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Lambda:  # pylint: disable=unidiomatic-typecheck
      return all((
          self.forward is other.forward,
          self.inverse is other.inverse,
          self.forward_log_det_jacobian is other.forward_log_det_jacobian,
          self.inverse_log_det_jacobian is other.inverse_log_det_jacobian,
          self.forward_and_log_det is other.forward_and_log_det,
          self.inverse_and_log_det is other.inverse_and_log_det,
      ))

    return False
