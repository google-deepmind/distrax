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
"""Gumbel bijector."""

from typing import Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion
import jax.numpy as jnp


Array = base.Array


class GumbelCDF(base.Bijector):
  """A bijector that computes the Gumbel CDF."""

  def __init__(self, loc=0., scale=1.):
    """Initializes a Gumbel bijector."""
    super().__init__(event_ndims_in=0)
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)

  def forward(self, x: Array) -> Array:   
    """Computes y = f(x)."""
    z = (x - self._loc) / self._scale
    return jnp.exp(-jnp.exp(z))

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    z = (x - self._loc) / self._scale
    return -z - jnp.exp(-z) - jnp.log(self._scale)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self.forward(x), self.forward_log_det_jacobian(x)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    x = self._loc - self._scale * jnp.log(-jnp.log(y))
    return x, jnp.log(self._scale / (-jnp.log(y) * y))
