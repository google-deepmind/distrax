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
