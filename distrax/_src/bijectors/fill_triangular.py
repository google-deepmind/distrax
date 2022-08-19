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
"""Fill triangular bijector."""

import jax.numpy as jnp
from typing import Tuple, Optional

from distrax._src.bijectors import bijector as base

Array = base.Array


class FillTriangular(base.Bijector):
  """A transformation that maps a vector to a triangular matrix. The triangular
   matrix can be either upper or lower triangular. By default, the lower
   triangular matrix is used.

  When projecting from a vector to a triangular matrix, entries of the matrix
   are populated row-wise. For example, if the vector is [1, 2, 3, 4, 5, 6],
   the triangular matrix will be:
  [[1, 0, 0],
  [2, 3, 0],
  [4, 5, 6]].
  """

  def __init__(
    self,
    matrix_shape: int,
    is_lower: Optional[bool] = True,
  ):
    """Initialise the `FillTriangular` bijector.

  Args:
  matrix_shape (int): The number of rows (or columns) in the original
  triangular matrix.
  upper (Optional[bool]): Whether or not the matrix being transformed
  is an upper or lower-triangular matrix. Defaults to True.
  """ """"""
    super().__init__(event_ndims_in=0)
    self.matrix_shape = matrix_shape
    self.index_fn = jnp.tril_indices if is_lower else jnp.triu_indices

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """The forward method maps from a vector to a triangular matrix.

    Args:
      x (Array): The 1-dimensional vector that is to be mapped into a
      triangular matrix.

    Returns:
      Tuple[Array, Array]: A triangular matrix and the log determinant of the
      Jacobian. The log-determinant here is just 0. as the bijection is simply
      reshaping.
    """
    # matrix_shape = jnp.sqrt(0.25 + 2. * jnp.shape(x)[0]) - 0.5
    # matrix_shape = jnp.asarray(matrix_shape).astype(jnp.int32)
    y = jnp.zeros((self.matrix_shape, self.matrix_shape))
    # Get the indexes for which we need to fill the triangular matrix
    idxs = self.index_fn(self.matrix_shape)
    # Fill the triangular matrix
    y = y.at[idxs].set(x)
    return y, jnp.array(0.0)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """The inverse method maps from a triangular matrix to a vector.

    Args:
      y (Array): The lower triangular

    Returns:
      Tuple[Array, Array]: The vectorised form of the supplied triangular
      matrix and the log determinant of the Jacobian. The log-determinant
      here is just 0. as the bijection is simply reshaping.
    """
    matrix_shape = y.shape[0]
    return y[self.index_fn(matrix_shape)], jnp.array(0.0)
