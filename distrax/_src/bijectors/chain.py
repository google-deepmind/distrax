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
"""Chain Bijector for composing a sequence of Bijector transformations."""

from typing import List, Sequence, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


class Chain(base.Bijector):
  """Composition of a sequence of bijectors into a single bijector.

  Bijectors are composable: if `f` and `g` are bijectors, then `g o f` is also
  a bijector. Given a sequence of bijectors `[f1, ..., fN]`, this class
  implements the bijector defined by `fN o ... o f1`.

  NOTE: the bijectors are applied in reverse order from the order they appear in
  the sequence. For example, consider the following code where `f` and `g` are
  two bijectors:
  ```
  layers = []
  layers.append(f)
  layers.append(g)
  bijector = distrax.Chain(layers)
  y = bijector.forward(x)
  ```
  The above code will transform `x` by first applying `g`, then `f`, so that
  `y = f(g(x))`.
  """

  def __init__(self, bijectors: Sequence[BijectorLike]):
    """Initializes a Chain bijector.

    Args:
      bijectors: a sequence of bijectors to be composed into one. Each bijector
        can be a distrax bijector, a TFP bijector, or a callable to be wrapped
        by `Lambda`. The sequence must contain at least one bijector.
    """
    if not bijectors:
      raise ValueError("The sequence of bijectors cannot be empty.")
    self._bijectors = [conversion.as_bijector(b) for b in bijectors]

    # Check that neighboring bijectors in the chain have compatible dimensions
    for i, (outer, inner) in enumerate(zip(self._bijectors[:-1],
                                           self._bijectors[1:])):
      if outer.event_ndims_in != inner.event_ndims_out:
        raise ValueError(
            f"The chain of bijector event shapes are incompatible. Bijector "
            f"{i} ({outer.name}) expects events with {outer.event_ndims_in} "
            f"dimensions, while Bijector {i+1} ({inner.name}) produces events "
            f"with {inner.event_ndims_out} dimensions.")

    is_constant_jacobian = all(b.is_constant_jacobian for b in self._bijectors)
    is_constant_log_det = all(b.is_constant_log_det for b in self._bijectors)
    super().__init__(
        event_ndims_in=self._bijectors[-1].event_ndims_in,
        event_ndims_out=self._bijectors[0].event_ndims_out,
        is_constant_jacobian=is_constant_jacobian,
        is_constant_log_det=is_constant_log_det)

  @property
  def bijectors(self) -> List[BijectorT]:
    """The list of bijectors in the chain."""
    return self._bijectors

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    for bijector in reversed(self._bijectors):
      x = bijector.forward(x)
    return x

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    for bijector in self._bijectors:
      y = bijector.inverse(y)
    return y

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    x, log_det = self._bijectors[-1].forward_and_log_det(x)
    for bijector in reversed(self._bijectors[:-1]):
      x, ld = bijector.forward_and_log_det(x)
      log_det += ld
    return x, log_det

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    y, log_det = self._bijectors[0].inverse_and_log_det(y)
    for bijector in self._bijectors[1:]:
      y, ld = bijector.inverse_and_log_det(y)
      log_det += ld
    return y, log_det

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Chain:  # pylint: disable=unidiomatic-typecheck
      if len(self.bijectors) != len(other.bijectors):
        return False
      for bij1, bij2 in zip(self.bijectors, other.bijectors):
        if not bij1.same_as(bij2):
          return False
      return True
    elif len(self.bijectors) == 1:
      return self.bijectors[0].same_as(other)

    return False

