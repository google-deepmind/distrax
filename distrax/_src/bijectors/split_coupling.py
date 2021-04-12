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
"""Split coupling bijector."""

from typing import Any, Callable, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import block
from distrax._src.utils import conversion
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class SplitCoupling(base.Bijector):
  """Split coupling bijector, with arbitrary conditioner & inner bijector.

  This coupling bijector splits the input array into two parts along a specified
  axis. One part remains unchanged, whereas the other part is transformed by an
  inner bijector conditioned on the unchanged part.

  Let `f` be a conditional bijector (the inner bijector) and `g` be a function
  (the conditioner). For `swap=False`, the split coupling bijector is defined as
  follows:

  - Forward:
    ```
    x = [x1, x2]
    y1 = x1
    y2 = f(x2; g(x1))
    y = [y1, y2]
    ```

  - Forward Jacobian log determinant:
    ```
    x = [x1, x2]
    log|det J(x)| = log|det df/dx2(x2; g(x1))|
    ```

  - Inverse:
    ```
    y = [y1, y2]
    x1 = y1
    x2 = f^{-1}(y2; g(y1))
    x = [x1, x2]
    ```

  - Inverse Jacobian log determinant:
    ```
    y = [y1, y2]
    log|det J(y)| = log|det df^{-1}/dy2(y2; g(y1))|
    ```

  Here, `[x1, x2]` is a partition of `x` along some axis. By default, `x1`
  remains unchanged and `x2` is transformed. If `swap=True`, `x2` will remain
  unchanged and `x1` will be transformed.
  """

  def __init__(self,
               split_index: int,
               event_ndims: int,
               conditioner: Callable[[Array], BijectorParams],
               bijector: Callable[[BijectorParams], base.BijectorLike],
               swap: bool = False,
               split_axis: int = -1):
    """Initializes a SplitCoupling bijector.

    Args:
      split_index: the index used to split the input. The input array will be
        split along the axis specified by `split_axis` into two parts. The first
        part will correspond to indices up to `split_index` (non-inclusive),
        whereas the second part will correspond to indices starting from
        `split_index` (inclusive).
      event_ndims: the number of event dimensions the bijector operates on. The
        `event_ndims_in` and `event_ndims_out` of the coupling bijector are both
        equal to `event_ndims`.
      conditioner: a function that computes the parameters of the inner bijector
        as a function of the unchanged part of the input. The output of the
        conditioner will be passed to `bijector` in order to obtain the inner
        bijector.
      bijector: a callable that returns the inner bijector that will be used to
        transform one of the two parts. The input to `bijector` is a set of
        parameters that can be used to configure the inner bijector. The
        `event_ndims_in` and `event_ndims_out` of the inner bijector must be
        equal, and less than or equal to `event_ndims`. If they are less than
        `event_ndims`, the remaining dimensions will be converted to event
        dimensions using `distrax.Block`.
      swap: by default, the part of the input up to `split_index` is the one
        that remains unchanged. If `swap` is True, then the other part remains
        unchanged and the first one is transformed instead.
      split_axis: the axis along which to split the input. Must be negative,
        that is, it must index from the end. By default, it's the last axis.
    """
    if split_index < 0:
      raise ValueError(
          f'The split index must be non-negative; got {split_index}.')
    if split_axis >= 0:
      raise ValueError(f'The split axis must be negative; got {split_axis}.')
    if event_ndims < 0:
      raise ValueError(
          f'`event_ndims` must be non-negative; got {event_ndims}.')
    if split_axis < -event_ndims:
      raise ValueError(
          f'The split axis points to an axis outside the event. With '
          f'`event_ndims == {event_ndims}`, the split axis must be between -1 '
          f'and {-event_ndims}. Got `split_axis == {split_axis}`.')
    self._split_index = split_index
    self._conditioner = conditioner
    self._bijector = bijector
    self._swap = swap
    self._split_axis = split_axis
    super().__init__(event_ndims_in=event_ndims)

  @property
  def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]:
    """The callable that returns the inner bijector of `SplitCoupling`."""
    return self._bijector

  @property
  def conditioner(self) -> Callable[[Array], BijectorParams]:
    """The conditioner function."""
    return self._conditioner

  @property
  def split_index(self) -> int:
    """The index used to split the input."""
    return self._split_index

  @property
  def swap(self) -> bool:
    """The flag that determines which part of the input remains unchanged."""
    return self._swap

  @property
  def split_axis(self) -> int:
    """The axis along which to split the input."""
    return self._split_axis

  def _split(self, x: Array) -> Tuple[Array, Array]:
    x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
    if self._swap:
      x1, x2 = x2, x1
    return x1, x2

  def _recombine(self, x1: Array, x2: Array) -> Array:
    if self._swap:
      x1, x2 = x2, x1
    return jnp.concatenate([x1, x2], self._split_axis)

  def _inner_bijector(self, params: BijectorParams) -> base.Bijector:
    """Returns an inner bijector for the passed params."""
    bijector = conversion.as_bijector(self._bijector(params))
    if bijector.event_ndims_in != bijector.event_ndims_out:
      raise ValueError(
          f'The inner bijector must have `event_ndims_in==event_ndims_out`. '
          f'Instead, it has `event_ndims_in=={bijector.event_ndims_in}` and '
          f'`event_ndims_out=={bijector.event_ndims_out}`.')
    extra_ndims = self.event_ndims_in - bijector.event_ndims_in
    if extra_ndims < 0:
      raise ValueError(
          f'The inner bijector can\'t have more event dimensions than the '
          f'coupling bijector. Got {bijector.event_ndims_in} for the inner '
          f'bijector and {self.event_ndims_in} for the coupling bijector.')
    elif extra_ndims > 0:
      bijector = block.Block(bijector, extra_ndims)
    return bijector

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    x1, x2 = self._split(x)
    params = self._conditioner(x1)
    y2, logdet = self._inner_bijector(params).forward_and_log_det(x2)
    return self._recombine(x1, y2), logdet

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    y1, y2 = self._split(y)
    params = self._conditioner(y1)
    x2, logdet = self._inner_bijector(params).inverse_and_log_det(y2)
    return self._recombine(y1, x2), logdet
