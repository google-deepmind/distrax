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
"""Wrapper to turn independent Bijectors into block Bijectors."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion
from distrax._src.utils import math

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


class Block(base.Bijector):
  """A wrapper that promotes a bijector to a block bijector.

  A block bijector applies a bijector to a k-dimensional array of events, but
  considers that array of events to be a single event. In practical terms, this
  means that the log det Jacobian will be summed over its last k dimensions.

  For example, consider a scalar bijector (such as `Tanh`) that operates on
  scalar events. We may want to apply this bijector identically to a 4D array of
  shape [N, H, W, C] representing a sequence of N images. Doing so naively will
  produce a log det Jacobian of shape [N, H, W, C], because the scalar bijector
  will assume scalar events and so all 4 dimensions will be considered as batch.
  To promote the scalar bijector to a "block scalar" that operates on the 3D
  arrays can be done by `Block(bijector, ndims=3)`. Then, applying the block
  bijector will produce a log det Jacobian of shape [N] as desired.

  In general, suppose `bijector` operates on n-dimensional events. Then,
  `Block(bijector, k)` will promote `bijector` to a block bijector that
  operates on (k + n)-dimensional events, summing the log det Jacobian over its
  last k dimensions. In practice, this means that the last k batch dimensions
  will be turned into event dimensions.
  """

  def __init__(self, bijector: BijectorLike, ndims: int):
    """Initializes a Block.

    Args:
      bijector: the bijector to be promoted to a block bijector. It can be a
        distrax bijector, a TFP bijector, or a callable to be wrapped by
        `Lambda`.
      ndims: number of batch dimensions to promote to event dimensions.
    """
    if ndims < 0:
      raise ValueError(f"`ndims` must be non-negative; got {ndims}.")
    self._bijector = conversion.as_bijector(bijector)
    self._ndims = ndims
    super().__init__(
        event_ndims_in=ndims + self._bijector.event_ndims_in,
        event_ndims_out=ndims + self._bijector.event_ndims_out,
        is_constant_jacobian=self._bijector.is_constant_jacobian,
        is_constant_log_det=self._bijector.is_constant_log_det)

  @property
  def bijector(self) -> BijectorT:
    """The base bijector, without promoting to a block bijector."""
    return self._bijector

  @property
  def ndims(self) -> int:
    """The number of batch dimensions promoted to event dimensions."""
    return self._ndims

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)
    return self._bijector.forward(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    self._check_inverse_input_shape(y)
    return self._bijector.inverse(y)

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    log_det = self._bijector.forward_log_det_jacobian(x)
    return math.sum_last(log_det, self._ndims)

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    log_det = self._bijector.inverse_log_det_jacobian(y)
    return math.sum_last(log_det, self._ndims)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    y, log_det = self._bijector.forward_and_log_det(x)
    return y, math.sum_last(log_det, self._ndims)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    x, log_det = self._bijector.inverse_and_log_det(y)
    return x, math.sum_last(log_det, self._ndims)

  @property
  def name(self) -> str:
    """Name of the bijector."""
    return self.__class__.__name__ + self._bijector.name

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Block:  # pylint: disable=unidiomatic-typecheck
      return self.bijector.same_as(other.bijector) and self.ndims == other.ndims

    return False
