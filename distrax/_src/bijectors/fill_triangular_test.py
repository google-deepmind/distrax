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
"""Tests for `fill_triangular.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import fill_triangular
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors

RTOL = 1e-5


class FillTriangularTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  def test_properties(self):
    bijector = fill_triangular.FillTriangular(matrix_shape=3)
    self.assertEqual(bijector.event_ndims_in, 0)
    self.assertEqual(bijector.event_ndims_out, 0)
    self.assertFalse(bijector.is_constant_jacobian)
    self.assertFalse(bijector.is_constant_log_det)

  @chex.all_variants
  @parameterized.parameters(True, False)
  def test_forward_method(self, is_lower):
    base_vector = jnp.array([1, 2, 3, 4, 5, 6])
    bijector = fill_triangular.FillTriangular(matrix_shape=3, is_lower=is_lower)
    x_triangular = self.variant(bijector.forward)(base_vector)
    if is_lower:
      self.assertTrue(jnp.sum(jnp.tril(x_triangular)) == jnp.sum(base_vector))
    elif not is_lower:
      self.assertTrue(jnp.sum(jnp.triu(x_triangular)) == jnp.sum(base_vector))

  @chex.all_variants
  @parameterized.parameters(True, False)
  def test_inverse_method(self, is_lower):
    random_array = jax.random.normal(self.seed, shape=(5, 5))
    psd_matrix = random_array @ random_array.T
    triangular_mat = jnp.linalg.cholesky(psd_matrix)
    if not is_lower:
      triangular_mat = triangular_mat.T

    bijector = fill_triangular.FillTriangular(matrix_shape=5, is_lower=is_lower)
    x_vector = self.variant(bijector.inverse)(triangular_mat)
    if is_lower:
      np.testing.assert_allclose(jnp.sum(jnp.tril(triangular_mat)),
                                 jnp.sum(x_vector),
                                 rtol=RTOL)
    elif not is_lower:
      np.testing.assert_allclose(jnp.sum(jnp.triu(triangular_mat)),
                                 jnp.sum(x_vector),
                                 rtol=RTOL)

  @chex.all_variants
  @parameterized.parameters(True, False)
  def test_inverse_log_jacobian(self, is_lower):
    random_array = jax.random.normal(self.seed, shape=(5, 5))
    psd_matrix = random_array @ random_array.T
    triangular_mat = jnp.linalg.cholesky(psd_matrix)
    if not is_lower:
      triangular_mat = triangular_mat.T

    bijector = fill_triangular.FillTriangular(matrix_shape=5, is_lower=is_lower)
    log_det_jac = self.variant(
        bijector.inverse_log_det_jacobian)(triangular_mat)
    self.assertTrue(log_det_jac == 0.0)

  @chex.all_variants
  @parameterized.parameters(True, False)
  def test_forward_log_jacobian(self, is_lower):
    base_vector = jnp.array([1, 2, 3, 4, 5, 6])
    bijector = fill_triangular.FillTriangular(matrix_shape=3, is_lower=is_lower)
    inv_log_det_jac = self.variant(
        bijector.forward_log_det_jacobian)(base_vector)
    self.assertTrue(inv_log_det_jac == 0.0)


if __name__ == "__main__":
  absltest.main()
