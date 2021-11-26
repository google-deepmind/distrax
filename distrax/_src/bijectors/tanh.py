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
"""Tanh bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
import jax
import jax.numpy as jnp


Array = base.Array


class Tanh(base.Bijector):
  """A bijector that computes the hyperbolic tangent.

  The log-determinant implementation in this bijector is more numerically stable
  than relying on the automatic differentiation approach used by Lambda, so this
  bijector should be preferred over Lambda(jnp.tanh) where possible. See
  `tfp.bijectors.Tanh` for details.

  When the absolute value of the input is large, `Tanh` becomes close to a
  constant, so that it is not possible to recover the input `x` from the output
  `y` within machine precision. In cases where it is needed to compute both the
  forward mapping and the backward mapping one after the other to recover the
  original input `x`, it is the user's responsibility to simplify the operation
  to avoid numerical issues; this is unlike the `tfp.bijectors.Tanh`. One
  example of such case is to use the bijector within a `Transformed`
  distribution and to obtain the log-probability of samples obtained from the
  distribution's `sample` method. For values of the samples for which it is not
  possible to apply the inverse bijector accurately, `log_prob` returns NaN.
  This can be avoided by using `sample_and_log_prob` instead of `sample`
  followed by `log_prob`.
  """

  def __init__(self):
    """Initializes a Tanh bijector."""
    super().__init__(event_ndims_in=0)

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    return 2 * (jnp.log(2) - x - jax.nn.softplus(-2 * x))

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return jnp.tanh(x), self.forward_log_det_jacobian(x)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    x = jnp.arctanh(y)
    return x, -self.forward_log_det_jacobian(x)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    return type(other) is Tanh  # pylint: disable=unidiomatic-typecheck
