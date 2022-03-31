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
"""Tests for `diag_plus_low_rank_linear.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors.diag_plus_low_rank_linear import DiagPlusLowRankLinear
from distrax._src.bijectors.tanh import Tanh
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class DiagPlusLowRankLinearTest(parameterized.TestCase):

  def test_static_properties(self):
    bij = DiagPlusLowRankLinear(
        diag=jnp.ones((4,)),
        u_matrix=jnp.ones((4, 2)),
        v_matrix=jnp.ones((4, 2)))
    self.assertTrue(bij.is_constant_jacobian)
    self.assertTrue(bij.is_constant_log_det)
    self.assertEqual(bij.event_ndims_in, 1)
    self.assertEqual(bij.event_ndims_out, 1)

  @parameterized.parameters(
      {'batch_shape': (), 'dtype': jnp.float16},
      {'batch_shape': (2, 3), 'dtype': jnp.float32},
  )
  def test_properties(self, batch_shape, dtype):
    bij = DiagPlusLowRankLinear(
        diag=jnp.ones(batch_shape + (4,), dtype),
        u_matrix=2. * jnp.ones(batch_shape + (4, 2), dtype),
        v_matrix=3. * jnp.ones(batch_shape + (4, 2), dtype))
    self.assertEqual(bij.event_dims, 4)
    self.assertEqual(bij.batch_shape, batch_shape)
    self.assertEqual(bij.dtype, dtype)
    self.assertEqual(bij.diag.shape, batch_shape + (4,))
    self.assertEqual(bij.u_matrix.shape, batch_shape + (4, 2))
    self.assertEqual(bij.v_matrix.shape, batch_shape + (4, 2))
    self.assertEqual(bij.matrix.shape, batch_shape + (4, 4))
    self.assertEqual(bij.diag.dtype, dtype)
    self.assertEqual(bij.u_matrix.dtype, dtype)
    self.assertEqual(bij.v_matrix.dtype, dtype)
    self.assertEqual(bij.matrix.dtype, dtype)
    np.testing.assert_allclose(bij.diag, 1., atol=1e-6)
    np.testing.assert_allclose(bij.u_matrix, 2., atol=1e-6)
    np.testing.assert_allclose(bij.v_matrix, 3., atol=1e-6)
    np.testing.assert_allclose(
        bij.matrix, np.tile(np.eye(4) + 12., batch_shape + (1, 1)), atol=1e-6)

  @parameterized.named_parameters(
      ('diag is 0d', {'diag': np.ones(()),
                      'u_matrix': np.ones((4, 2)),
                      'v_matrix': np.ones((4, 2))}),
      ('u_matrix is 1d', {'diag': np.ones((4,)),
                          'u_matrix': np.ones((4,)),
                          'v_matrix': np.ones((4, 2))}),
      ('v_matrix is 1d', {'diag': np.ones((4,)),
                          'u_matrix': np.ones((4, 2)),
                          'v_matrix': np.ones((4,))}),
      ('diag has wrong dim', {'diag': np.ones((3,)),
                              'u_matrix': np.ones((4, 2)),
                              'v_matrix': np.ones((4, 2))}),
      ('u_matrix has wrong dim', {'diag': np.ones((4,)),
                                  'u_matrix': np.ones((3, 2)),
                                  'v_matrix': np.ones((4, 2))}),
      ('v_matrix has wrong dim', {'diag': np.ones((4,)),
                                  'u_matrix': np.ones((4, 2)),
                                  'v_matrix': np.ones((3, 2))}),
  )
  def test_raises_with_invalid_parameters(self, params):
    with self.assertRaises(ValueError):
      DiagPlusLowRankLinear(**params)

  @chex.all_variants
  @parameterized.parameters(
      ((5,), (5,), (5,), (5,)),
      ((5,), (), (), ()),
      ((), (5,), (), ()),
      ((), (), (5,), ()),
      ((), (), (), (5,)),
  )
  def test_batched_parameters(self, diag_batch_shape, u_matrix_batch_shape,
                              v_matrix_batch_shape, input_batch_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = jax.random.uniform(next(prng), diag_batch_shape + (4,)) + 0.5
    u_matrix = jax.random.uniform(next(prng), u_matrix_batch_shape + (4, 1))
    v_matrix = jax.random.uniform(next(prng), v_matrix_batch_shape + (4, 1))
    bij = DiagPlusLowRankLinear(diag, u_matrix, v_matrix)

    x = jax.random.normal(next(prng), input_batch_shape + (4,))
    y, logdet_fwd = self.variant(bij.forward_and_log_det)(x)
    z, logdet_inv = self.variant(bij.inverse_and_log_det)(x)

    output_batch_shape = jnp.broadcast_shapes(
        diag_batch_shape, u_matrix_batch_shape, v_matrix_batch_shape,
        input_batch_shape)

    self.assertEqual(y.shape, output_batch_shape + (4,))
    self.assertEqual(z.shape, output_batch_shape + (4,))
    self.assertEqual(logdet_fwd.shape, output_batch_shape)
    self.assertEqual(logdet_inv.shape, output_batch_shape)

    diag = jnp.broadcast_to(diag, output_batch_shape + (4,)).reshape((-1, 4))
    u_matrix = jnp.broadcast_to(
        u_matrix, output_batch_shape + (4, 1)).reshape((-1, 4, 1))
    v_matrix = jnp.broadcast_to(
        v_matrix, output_batch_shape + (4, 1)).reshape((-1, 4, 1))
    x = jnp.broadcast_to(x, output_batch_shape + (4,)).reshape((-1, 4))
    y = y.reshape((-1, 4))
    z = z.reshape((-1, 4))
    logdet_fwd = logdet_fwd.flatten()
    logdet_inv = logdet_inv.flatten()

    for i in range(np.prod(output_batch_shape)):
      bij = DiagPlusLowRankLinear(diag[i], u_matrix[i], v_matrix[i])
      this_y, this_logdet_fwd = self.variant(bij.forward_and_log_det)(x[i])
      this_z, this_logdet_inv = self.variant(bij.inverse_and_log_det)(x[i])
      np.testing.assert_allclose(this_y, y[i], atol=1e-6)
      np.testing.assert_allclose(this_z, z[i], atol=1e-6)
      np.testing.assert_allclose(this_logdet_fwd, logdet_fwd[i], atol=1e-6)
      np.testing.assert_allclose(this_logdet_inv, logdet_inv[i], atol=1e-6)

  @chex.all_variants
  @parameterized.parameters(
      {'batch_shape': (), 'param_shape': ()},
      {'batch_shape': (2, 3), 'param_shape': (3,)},
  )
  def test_identity_initialization(self, batch_shape, param_shape):
    bij = DiagPlusLowRankLinear(
        diag=jnp.ones(param_shape + (4,)),
        u_matrix=jnp.zeros(param_shape + (4, 1)),
        v_matrix=jnp.zeros(param_shape + (4, 1)))
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    x = jax.random.normal(next(prng), batch_shape + (4,))

    # Forward methods.
    y, logdet = self.variant(bij.forward_and_log_det)(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(logdet, jnp.zeros(batch_shape))

    # Inverse methods.
    x_rec, logdet = self.variant(bij.inverse_and_log_det)(y)
    np.testing.assert_array_equal(x_rec, y)
    np.testing.assert_array_equal(logdet, jnp.zeros(batch_shape))

  @chex.all_variants
  @parameterized.parameters(
      {'batch_shape': (), 'param_shape': ()},
      {'batch_shape': (2, 3), 'param_shape': (3,)}
  )
  def test_inverse_methods(self, batch_shape, param_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = jax.random.uniform(next(prng), param_shape + (4,)) + 0.5
    u_matrix = jax.random.uniform(next(prng), param_shape + (4, 1))
    v_matrix = jax.random.uniform(next(prng), param_shape + (4, 1))
    bij = DiagPlusLowRankLinear(diag, u_matrix, v_matrix)
    x = jax.random.normal(next(prng), batch_shape + (4,))
    y, logdet_fwd = self.variant(bij.forward_and_log_det)(x)
    x_rec, logdet_inv = self.variant(bij.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_rec, x, atol=1e-6)
    np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)

  @chex.all_variants
  def test_forward_jacobian_det(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = jax.random.uniform(next(prng), (4,)) + 0.5
    u_matrix = jax.random.uniform(next(prng), (4, 1))
    v_matrix = jax.random.uniform(next(prng), (4, 1))
    bij = DiagPlusLowRankLinear(diag, u_matrix, v_matrix)

    batched_x = jax.random.normal(next(prng), (10, 4))
    single_x = jax.random.normal(next(prng), (4,))
    batched_logdet = self.variant(bij.forward_log_det_jacobian)(batched_x)

    jacobian_fn = jax.jacfwd(bij.forward)
    logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_x))[1]
    for logdet in batched_logdet:
      np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-4)

  @chex.all_variants
  def test_inverse_jacobian_det(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = jax.random.uniform(next(prng), (4,)) + 0.5
    u_matrix = jax.random.uniform(next(prng), (4, 1))
    v_matrix = jax.random.uniform(next(prng), (4, 1))
    bij = DiagPlusLowRankLinear(diag, u_matrix, v_matrix)

    batched_y = jax.random.normal(next(prng), (10, 4))
    single_y = jax.random.normal(next(prng), (4,))
    batched_logdet = self.variant(bij.inverse_log_det_jacobian)(batched_y)

    jacobian_fn = jax.jacfwd(bij.inverse)
    logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_y))[1]
    for logdet in batched_logdet:
      np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-4)

  def test_raises_on_invalid_input_shape(self):
    bij = DiagPlusLowRankLinear(
        diag=jnp.ones((4,)),
        u_matrix=jnp.ones((4, 2)),
        v_matrix=jnp.ones((4, 2)))
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.subTest(fn=fn):
        with self.assertRaises(ValueError):
          fn(jnp.array(0))

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bij = DiagPlusLowRankLinear(
        diag=jnp.ones((4,)),
        u_matrix=jnp.ones((4, 2)),
        v_matrix=jnp.ones((4, 2)))
    x = np.zeros((4,))
    f(x, bij)

  def test_same_as_itself(self):
    bij = DiagPlusLowRankLinear(
        diag=jnp.ones((4,)),
        u_matrix=jnp.ones((4, 2)),
        v_matrix=jnp.ones((4, 2)))
    self.assertTrue(bij.same_as(bij))

  def test_not_same_as_others(self):
    bij = DiagPlusLowRankLinear(
        diag=jnp.ones((4,)),
        u_matrix=jnp.ones((4, 2)),
        v_matrix=jnp.ones((4, 2)))
    other = DiagPlusLowRankLinear(
        diag=2. * jnp.ones((4,)),
        u_matrix=jnp.ones((4, 2)),
        v_matrix=jnp.ones((4, 2)))
    self.assertFalse(bij.same_as(other))
    self.assertFalse(bij.same_as(Tanh()))


if __name__ == '__main__':
  absltest.main()
