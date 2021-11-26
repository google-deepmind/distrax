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
"""Wrapper for inverting a Distrax Bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


class Inverse(base.Bijector):
  """A bijector that inverts a given bijector.

  That is, if `bijector` implements the transformation `f`, `Inverse(bijector)`
  implements the inverse transformation `f^{-1}`.

  The inversion is performed by swapping the forward with the corresponding
  inverse methods of the given bijector.
  """

  def __init__(self, bijector: BijectorLike):
    """Initializes an Inverse bijector.

    Args:
      bijector: the bijector to be inverted. It can be a distrax bijector, a TFP
        bijector, or a callable to be wrapped by `Lambda`.
    """
    self._bijector = conversion.as_bijector(bijector)
    super().__init__(
        event_ndims_in=self._bijector.event_ndims_out,
        event_ndims_out=self._bijector.event_ndims_in,
        is_constant_jacobian=self._bijector.is_constant_jacobian,
        is_constant_log_det=self._bijector.is_constant_log_det)

  @property
  def bijector(self) -> BijectorT:
    """The base bijector that was the input to `Inverse`."""
    return self._bijector

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    return self._bijector.inverse(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return self._bijector.forward(y)

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    return self._bijector.inverse_log_det_jacobian(x)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return self._bijector.forward_log_det_jacobian(y)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self._bijector.inverse_and_log_det(x)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self._bijector.forward_and_log_det(y)

  @property
  def name(self) -> str:
    """Name of the bijector."""
    return self.__class__.__name__ + self._bijector.name

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Inverse:  # pylint: disable=unidiomatic-typecheck
      return self.bijector.same_as(other.bijector)
    return False
