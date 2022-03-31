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
"""Linear bijector."""

import abc
from typing import Sequence, Tuple

from distrax._src.bijectors import bijector as base
import jax.numpy as jnp

Array = base.Array


class Linear(base.Bijector, metaclass=abc.ABCMeta):
  """Base class for linear bijectors.

  This class provides a base class for bijectors defined as `f(x) = Ax`,
  where `A` is a `DxD` matrix and `x` is a `D`-dimensional vector.
  """

  def __init__(self,
               event_dims: int,
               batch_shape: Sequence[int],
               dtype: jnp.dtype):
    """Initializes a `Linear` bijector.

    Args:
      event_dims: the dimensionality `D` of the event `x`. It is assumed that
        `x` is a vector of length `event_dims`.
      batch_shape: the batch shape of the bijector.
      dtype: the data type of matrix `A`.
    """
    super().__init__(event_ndims_in=1, is_constant_jacobian=True)
    self._event_dims = event_dims
    self._batch_shape = tuple(batch_shape)
    self._dtype = dtype

  @property
  def matrix(self) -> Array:
    """The matrix `A` of the transformation.

    To be optionally implemented in a subclass.

    Returns:
      An array of shape `batch_shape + (event_dims, event_dims)` and data type
      `dtype`.
    """
    raise NotImplementedError(
        f"Linear bijector {self.name} does not implement `matrix`.")

  @property
  def event_dims(self) -> int:
    """The dimensionality `D` of the event `x`."""
    return self._event_dims

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """The batch shape of the bijector."""
    return self._batch_shape

  @property
  def dtype(self) -> jnp.dtype:
    """The data type of matrix `A`."""
    return self._dtype
