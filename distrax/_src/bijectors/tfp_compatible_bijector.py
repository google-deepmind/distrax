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
"""Wrapper to adapt a Distrax bijector for use in TFP."""

from typing import Any, Optional, Tuple

import chex
from distrax._src.bijectors import bijector
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

Array = chex.Array
Bijector = bijector.Bijector
TangentSpace = tfp.experimental.tangent_spaces.TangentSpace


def tfp_compatible_bijector(
    base_bijector: Bijector,
    name: Optional[str] = None):
  """Create a TFP-compatible bijector from a Distrax bijector.

  Given a Distrax bijector, return a wrapped bijector that behaves as a TFP
  bijector, to be used in TFP meta-bijectors and the TransformedDistribution.
  In particular, the wrapped bijector implements the methods
  `[forward|inverse]_event_ndims`, `[forward|inverse]_event_shape`,
  `[forward|inverse]_event_shape_tensor`, `[forward|inverse]_log_det_jacobian`,
  and the properties `[forward|inverse]_min_event_ndims`. Other attributes are
  delegated to the `base_bijector`.

  The methods of the resulting object do not take a `name` argument,
  unlike their TFP equivalents.

  The `shape` methods are implemented by tracing the `forward` and `inverse`
  methods of the bijector, applied to a zero tensor of the requested dtype. If
  the `forward` or `inverse` methods are not traceable or cannot be applied to a
  zero tensor, then we cannot guarantee the correctness of the result.

  Args:
    base_bijector: A Distrax bijector.
    name: The bijector name.

  Returns:
    An object that behaves like a TFP bijector.
  """

  name_ = name

  class TFPCompatibleBijector(base_bijector.__class__):
    """Class to wrap a Distrax bijector."""

    def __init__(self):
      self._is_injective = True
      self._is_permutation = False
      self._parts_interact = False

      self.dtype = None
      self.has_static_min_event_ndims = True
      self.forward_min_event_ndims = base_bijector.event_ndims_in
      self.inverse_min_event_ndims = base_bijector.event_ndims_out

    def __getattr__(self, name: str):
      return getattr(base_bijector, name)

    def forward_and_log_det(self, x: Array) -> Array:
      """See `Bijector.forward_and_log_det`."""
      return base_bijector.forward_and_log_det(x)

    @property
    def name(self) -> str:
      """The name of the wrapped bijector."""
      return name_ or f"TFPCompatible{base_bijector.name}"

    def experimental_batch_shape(self, x_event_ndims=None, y_event_ndims=None):
      raise NotImplementedError()

    def experimental_batch_shape_tensor(
        self, x_event_ndims=None, y_event_ndims=None):
      raise NotImplementedError()

    def forward_dtype(self, _: jnp.dtype) -> None:
      """Returns None, making no promise regarding dtypes."""
      return None

    def inverse_dtype(self, _: jnp.dtype) -> None:
      """Returns None, making no promise regarding dtypes."""
      return None

    def forward_event_ndims(self, event_ndims: int) -> int:
      """Returns the number of event dimensions of the output of `forward`."""
      extra_event_ndims = self._check_ndims(
          "Forward", event_ndims, base_bijector.event_ndims_in)
      return base_bijector.event_ndims_out + extra_event_ndims

    def inverse_event_ndims(self, event_ndims: int) -> int:
      """Returns the number of event dimensions of the output of `inverse`."""
      extra_event_ndims = self._check_ndims(
          "Inverse", event_ndims, base_bijector.event_ndims_out)
      return base_bijector.event_ndims_in + extra_event_ndims

    def forward_event_shape(self, event_shape) -> tfp.tf2jax.TensorShape:
      """Returns the shape of the output of `forward` as a `TensorShape`."""
      self._check_shape("Forward", event_shape, base_bijector.event_ndims_in)
      forward_event_shape = jax.eval_shape(
          base_bijector.forward, jnp.zeros(event_shape)).shape
      return tfp.tf2jax.TensorShape(forward_event_shape)

    def inverse_event_shape(self, event_shape) -> tfp.tf2jax.TensorShape:
      """Returns the shape of the output of `inverse` as a `TensorShape`."""
      self._check_shape("Inverse", event_shape, base_bijector.event_ndims_out)
      inverse_event_shape = jax.eval_shape(
          base_bijector.inverse, jnp.zeros(event_shape)).shape
      return tfp.tf2jax.TensorShape(inverse_event_shape)

    def forward_event_shape_tensor(self, event_shape) -> Array:
      """Returns the shape of the output of `forward` as a `jnp.array`."""
      self._check_shape("Forward", event_shape, base_bijector.event_ndims_in)
      forward_event_shape = jax.eval_shape(
          base_bijector.forward, jnp.zeros(event_shape)).shape
      return jnp.array(forward_event_shape, dtype=jnp.int32)

    def inverse_event_shape_tensor(self, event_shape) -> Array:
      """Returns the shape of the output of `inverse` as a `jnp.array`."""
      self._check_shape("Inverse", event_shape, base_bijector.event_ndims_out)
      inverse_event_shape = jax.eval_shape(
          base_bijector.inverse, jnp.zeros(event_shape)).shape
      return jnp.array(inverse_event_shape, dtype=jnp.int32)

    def forward_log_det_jacobian(
        self, x: Array, event_ndims: Optional[int] = None) -> Array:
      """See `Bijector.forward_log_det_jacobian`."""
      extra_event_ndims = self._check_ndims(
          "Forward", event_ndims, base_bijector.event_ndims_in)
      fldj = base_bijector.forward_log_det_jacobian(x)
      return math.sum_last(fldj, extra_event_ndims)

    def inverse_log_det_jacobian(
        self, y: Array, event_ndims: Optional[int] = None) -> Array:
      """See `Bijector.inverse_log_det_jacobian`."""
      extra_event_ndims = self._check_ndims(
          "Inverse", event_ndims, base_bijector.event_ndims_out)
      ildj = base_bijector.inverse_log_det_jacobian(y)
      return math.sum_last(ildj, extra_event_ndims)

    def _check_ndims(
        self, direction: str, event_ndims: int, expected_ndims: int) -> int:
      """Checks that `event_ndims` are correct and returns any extra ndims."""
      if event_ndims is not None and event_ndims < expected_ndims:
        raise ValueError(f"{direction} `event_ndims` of {self.name} must be at "
                         f"least {expected_ndims} but was passed {event_ndims} "
                         f"instead.")
      return 0 if event_ndims is None else event_ndims - expected_ndims

    def _check_shape(
        self, direction: str, event_shape: Any, expected_ndims: int):
      """Checks that `event_shape` is correct, raising ValueError otherwise."""
      if len(event_shape) < expected_ndims:
        raise ValueError(f"{direction} `event_shape` of {self.name} must have "
                         f"at least {expected_ndims} dimensions, but was "
                         f"{event_shape} which has only {len(event_shape)} "
                         f"dimensions instead.")

    def experimental_compute_density_correction(
        self,
        x: Array,
        tangent_space: TangentSpace,
        backward_compat: bool = True,
        **kwargs) -> Tuple[Array, TangentSpace]:
      """Density correction for this transform wrt the tangent space, at x.

      See `tfp.bijectors.experimental_compute_density_correction`, and
      Radul and Alexeev, AISTATS 2021, “The Base Measure Problem and its
      Solution”, https://arxiv.org/abs/2010.09647.

      Args:
        x: `float` or `double` `Array`.
        tangent_space: `TangentSpace` or one of its subclasses.  The tangent to
          the support manifold at `x`.
        backward_compat: unused
        **kwargs: Optional keyword arguments forwarded to tangent space methods.

      Returns:
        density_correction: `Array` representing the density correction---in log
          space---under the transformation that this Bijector denotes. Assumes
          the Bijector is dimension-preserving.
        space: `TangentSpace` representing the new tangent to the support
          manifold, at `x`.
      """
      del backward_compat
      # We ignore the `backward_compat` flag and always act as though it's
      # true because Distrax bijectors and distributions need not follow the
      # base measure protocol from TFP. This implies that we expect to return
      # the `FullSpace` tangent space.
      return tangent_space.transform_dimension_preserving(x, self, **kwargs)

  return TFPCompatibleBijector()
