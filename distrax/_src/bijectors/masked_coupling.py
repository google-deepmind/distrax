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
"""Masked coupling bijector."""

from typing import Any, Callable, Optional, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class MaskedCoupling(base.Bijector):
  """Coupling bijector that uses a mask to specify which inputs are transformed.

  This coupling bijector takes in a boolean mask that indicates which inputs are
  transformed. Inputs where `mask==True` remain unchanged. Inputs where
  `mask==False` are transformed by an inner bijector, conditioned on the masked
  inputs.

  The number of event dimensions this bijector operates on is referred to as
  `event_ndims`, and is equal to both `event_ndims_in` and `event_ndims_out`.
  By default, `event_ndims` is equal to `mask.ndim + inner_event_ndims`. The
  user can override this by passing an explicit value for `event_ndims`. If
  `event_ndims > mask.ndim + inner_event_ndims`, the mask is broadcast to the
  extra dimensions. If `event_ndims < mask.ndims + inner_event_ndims`, the mask
  is assumed to be a batch of masks that will broadcast against the input.

  Let `f` be a conditional bijector (the inner bijector), `g` be a function (the
  conditioner), and `m` be a boolean mask interpreted numerically, such that
  True is 1 and False is 0. The masked coupling bijector is defined as follows:

  - Forward: `y = (1-m) * f(x; g(m*x)) + m*x`

  - Forward Jacobian log determinant:
    `log|det J(x)| = sum((1-m) * log|df/dx(x; g(m*x))|)`

  - Inverse: `x = (1-m) * f^{-1}(y; g(m*y)) + m*y`

  - Inverse Jacobian log determinant:
    `log|det J(y)| = sum((1-m) * log|df^{-1}/dy(y; g(m*y))|)`
  """

  def __init__(self,
               mask: Array,
               conditioner: Callable[[Array], BijectorParams],
               bijector: Callable[[BijectorParams], base.BijectorLike],
               event_ndims: Optional[int] = None,
               inner_event_ndims: int = 0):
    """Initializes a MaskedCoupling bijector.

    Args:
      mask: the mask, or a batch of masks. Its elements must be boolean; a value
        of True indicates that the corresponding input remains unchanged, and a
        value of False indicates that the corresponding input is transformed.
        The mask should have `mask.ndim` equal to the number of batch dimensions
        plus `event_ndims - inner_event_ndims`. In particular, an inner event
        is either fully masked or fully un-masked: it is not possible to be
        partially masked.
      conditioner: a function that computes the parameters of the inner bijector
        as a function of the masked input. The output of the conditioner will be
        passed to `bijector` in order to obtain the inner bijector.
      bijector: a callable that returns the inner bijector that will be used to
        transform the input. The input to `bijector` is a set of parameters that
        can be used to configure the inner bijector. The `event_ndims_in` and
        `event_ndims_out` of this bijector must match the `inner_event_dims`.
        For example, if `inner_event_dims` is `0`, then the inner bijector must
        be a scalar bijector.
      event_ndims: the number of array dimensions the bijector operates on. If
        None, it defaults to `mask.ndim + inner_event_ndims`. Both
        `event_ndims_in` and `event_ndims_out` are equal to `event_ndims`. Note
        that `event_ndims` should be at least as large as `inner_event_ndims`.
      inner_event_ndims: the number of array dimensions the inner bijector
        operates on. This is `0` by default, meaning the inner bijector acts on
        scalars.
    """
    if mask.dtype != bool:
      raise ValueError(f'`mask` must have values of type `bool`; got values of'
                       f' type `{mask.dtype}`.')
    if event_ndims is not None and event_ndims < inner_event_ndims:
      raise ValueError(f'`event_ndims={event_ndims}` should be at least as'
                       f' large as `inner_event_ndims={inner_event_ndims}`.')
    self._mask = mask
    self._event_mask = jnp.reshape(mask, mask.shape + (1,) * inner_event_ndims)
    self._conditioner = conditioner
    self._bijector = bijector
    self._inner_event_ndims = inner_event_ndims
    if event_ndims is None:
      self._event_ndims = mask.ndim + inner_event_ndims
    else:
      self._event_ndims = event_ndims
    super().__init__(event_ndims_in=self._event_ndims)

  @property
  def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]:
    """The callable that returns the inner bijector of `MaskedCoupling`."""
    return self._bijector

  @property
  def conditioner(self) -> Callable[[Array], BijectorParams]:
    """The conditioner function."""
    return self._conditioner

  @property
  def mask(self) -> Array:
    """The mask characterizing the `MaskedCoupling`, with boolean `dtype`."""
    return self._mask

  def _inner_bijector(self, params: BijectorParams) -> base.Bijector:
    bijector = conversion.as_bijector(self._bijector(params))
    if (bijector.event_ndims_in != self._inner_event_ndims
        or bijector.event_ndims_out != self._inner_event_ndims):
      raise ValueError(
          'The inner bijector event ndims in and out must match the'
          f' `inner_event_ndims={self._inner_event_ndims}`. Instead, got'
          f' `event_ndims_in={bijector.event_ndims_in}` and'
          f' `event_ndims_out={bijector.event_ndims_out}`.')
    return bijector

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    masked_x = jnp.where(self._event_mask, x, 0.)
    params = self._conditioner(masked_x)
    y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
    y = jnp.where(self._event_mask, x, y0)
    logdet = math.sum_last(
        jnp.where(self._mask, 0., log_d),
        self._event_ndims - self._inner_event_ndims)
    return y, logdet

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    masked_y = jnp.where(self._event_mask, y, 0.)
    params = self._conditioner(masked_y)
    x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
    x = jnp.where(self._event_mask, y, x0)
    logdet = math.sum_last(jnp.where(self._mask, 0., log_d),
                           self._event_ndims - self._inner_event_ndims)
    return x, logdet
