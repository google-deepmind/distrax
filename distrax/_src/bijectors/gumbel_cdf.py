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
"""GumbelCDF bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
import jax.numpy as jnp

Array = base.Array


class GumbelCDF(base.Bijector):
  """A bijector that computes the Gumbel cumulative density function (CDF).

  The Gumbel CDF is given by `y = f(x) = exp(-exp(-x))` for a scalar input `x`.
  Its inverse is `x = -log(-log(y))`. The log-det Jacobian of the transformation
  is `log df/dx = -exp(-x) - x`.
  """

  def __init__(self):
    """Initializes a GumbelCDF bijector."""
    super().__init__(event_ndims_in=0)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    exp_neg_x = jnp.exp(-x)
    y = jnp.exp(-exp_neg_x)
    log_det = - x - exp_neg_x
    return y, log_det

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    log_y = jnp.log(y)
    x = -jnp.log(-log_y)
    return x, x - log_y

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    return type(other) is GumbelCDF  # pylint: disable=unidiomatic-typecheck
