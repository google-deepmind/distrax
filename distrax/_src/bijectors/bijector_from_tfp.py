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
"""Distrax adapter for Bijectors from TensorFlow Probability."""

from typing import Callable, Tuple

from distrax._src.bijectors import bijector as base
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfb = tfp.bijectors

Array = base.Array


class BijectorFromTFP(base.Bijector):
  """Wrapper around a TFP bijector that turns it into a Distrax bijector.

  TFP bijectors and Distrax bijectors have similar but not identical semantics,
  which makes them not directly compatible. This wrapper guarantees that the
  wrapepd TFP bijector fully satisfies the semantics of Distrax, which enables
  any TFP bijector to be used by Distrax without modification.
  """

  def __init__(self, tfp_bijector: tfb.Bijector):
    """Initializes a BijectorFromTFP.

    Args:
      tfp_bijector: TFP bijector to convert to Distrax bijector.
    """
    self._tfp_bijector = tfp_bijector
    super().__init__(
        event_ndims_in=tfp_bijector.forward_min_event_ndims,
        event_ndims_out=tfp_bijector.inverse_min_event_ndims,
        is_constant_jacobian=tfp_bijector.is_constant_jacobian)

  def __getattr__(self, name: str):
    return getattr(self._tfp_bijector, name)

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    return self._tfp_bijector.forward(x)

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return self._tfp_bijector.inverse(y)

  def _ensure_batch_shape(self,
                          logdet: Array,
                          event_ndims_out: int,
                          forward_fn: Callable[[Array], Array],
                          x: Array) -> Array:
    """Broadcasts logdet to the batch shape as required."""
    if self._tfp_bijector.is_constant_jacobian:
      # If the Jacobian is constant, TFP may return a log det that doesn't have
      # full batch shape, but is broadcastable to it. Distrax assumes that the
      # log det is always batch-shaped, so we broadcast.
      y_shape = jax.eval_shape(forward_fn, x).shape
      if event_ndims_out == 0:
        batch_shape = y_shape
      else:
        batch_shape = y_shape[:-event_ndims_out]
      logdet = jnp.broadcast_to(logdet, batch_shape)
    return logdet

  def forward_log_det_jacobian(self, x: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    logdet = self._tfp_bijector.forward_log_det_jacobian(x, self.event_ndims_in)
    logdet = self._ensure_batch_shape(
        logdet, self.event_ndims_out, self._tfp_bijector.forward, x)
    return logdet

  def inverse_log_det_jacobian(self, y: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    logdet = self._tfp_bijector.inverse_log_det_jacobian(
        y, self.event_ndims_out)
    logdet = self._ensure_batch_shape(
        logdet, self.event_ndims_in, self._tfp_bijector.inverse, y)
    return logdet

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    y = self._tfp_bijector.forward(x)
    logdet = self._tfp_bijector.forward_log_det_jacobian(x, self.event_ndims_in)
    logdet = self._ensure_batch_shape(
        logdet, self.event_ndims_out, self._tfp_bijector.forward, x)
    return y, logdet

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    x = self._tfp_bijector.inverse(y)
    logdet = self._tfp_bijector.inverse_log_det_jacobian(
        y, self.event_ndims_out)
    logdet = self._ensure_batch_shape(
        logdet, self.event_ndims_in, self._tfp_bijector.inverse, y)
    return x, logdet

  @property
  def name(self) -> str:
    """Name of the bijector."""
    return self._tfp_bijector.name
