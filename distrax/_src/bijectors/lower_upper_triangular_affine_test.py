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
"""Tests for `lower_upper_triangular_affine.py`."""

import copy

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors.lower_upper_triangular_affine import LowerUpperTriangularAffine
from distrax._src.bijectors.tanh import Tanh
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class LowerUpperTriangularAffineTest(parameterized.TestCase):

  def test_jacobian_is_constant_property(self):
    bijector = LowerUpperTriangularAffine(
        matrix=jnp.eye(4), bias=jnp.zeros((4,)))
    self.assertTrue(bijector.is_constant_jacobian)
    self.assertTrue(bijector.is_constant_log_det)

  def test_properties(self):
    bijector = LowerUpperTriangularAffine(
        matrix=jnp.array([[2., 3.], [4., 5.]]),
        bias=jnp.ones((2,)))
    lower = np.array([[1., 0.], [4., 1.]])
    upper = np.array([[2., 3.], [0., 5.]])
    np.testing.assert_allclose(bijector.lower, lower, atol=1e-6)
    np.testing.assert_allclose(bijector.upper, upper, atol=1e-6)
    np.testing.assert_allclose(bijector.matrix, lower @ upper, atol=1e-6)
    np.testing.assert_allclose(bijector.bias, np.ones((2,)), atol=1e-6)

  @parameterized.named_parameters(
      ('matrix is 0d', {'matrix': np.zeros(()), 'bias': np.zeros((4,))}),
      ('matrix is 1d', {'matrix': np.zeros((4,)), 'bias': np.zeros((4,))}),
      ('bias is 0d', {'matrix': np.zeros((4, 4)), 'bias': np.zeros(())}),
      ('matrix is not square',
       {'matrix': np.zeros((3, 4)), 'bias': np.zeros((4,))}),
      ('matrix and bias shapes do not agree',
       {'matrix': np.zeros((4, 4)), 'bias': np.zeros((3,))}),
  )
  def test_raises_with_invalid_parameters(self, bij_params):
    with self.assertRaises(ValueError):
      LowerUpperTriangularAffine(**bij_params)

  @chex.all_variants
  @parameterized.parameters(
      ((5,), (5,), (5,)),
      ((5,), (5,), ()),
      ((5,), (), (5,)),
      ((), (5,), (5,)),
      ((), (), (5,)),
      ((), (5,), ()),
      ((5,), (), ()),
  )
  def test_batched_parameters(self, matrix_batch_shape, bias_batch_shape,
                              input_batch_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    matrix = jax.random.uniform(
        next(prng), matrix_batch_shape + (4, 4)) + jnp.eye(4)
    bias = jax.random.normal(next(prng), bias_batch_shape + (4,))
    bijector = LowerUpperTriangularAffine(matrix, bias)

    x = jax.random.normal(next(prng), input_batch_shape + (4,))
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    z, logdet_inv = self.variant(bijector.inverse_and_log_det)(x)

    output_batch_shape = jnp.broadcast_arrays(
        matrix[..., 0, 0], bias[..., 0], x[..., 0])[0].shape

    self.assertEqual(y.shape, output_batch_shape + (4,))
    self.assertEqual(z.shape, output_batch_shape + (4,))
    self.assertEqual(logdet_fwd.shape, output_batch_shape)
    self.assertEqual(logdet_inv.shape, output_batch_shape)

    matrix = jnp.broadcast_to(
        matrix, output_batch_shape + (4, 4)).reshape((-1, 4, 4))
    bias = jnp.broadcast_to(bias, output_batch_shape + (4,)).reshape((-1, 4))
    x = jnp.broadcast_to(x, output_batch_shape + (4,)).reshape((-1, 4))
    y = y.reshape((-1, 4))
    z = z.reshape((-1, 4))
    logdet_fwd = logdet_fwd.flatten()
    logdet_inv = logdet_inv.flatten()

    for i in range(np.prod(output_batch_shape)):
      bijector = LowerUpperTriangularAffine(matrix[i], bias[i])
      this_y, this_logdet_fwd = self.variant(bijector.forward_and_log_det)(x[i])
      this_z, this_logdet_inv = self.variant(bijector.inverse_and_log_det)(x[i])
      np.testing.assert_allclose(this_y, y[i], atol=9e-3)
      np.testing.assert_allclose(this_z, z[i], atol=7e-6)
      np.testing.assert_allclose(this_logdet_fwd, logdet_fwd[i], atol=1e-7)
      np.testing.assert_allclose(this_logdet_inv, logdet_inv[i], atol=7e-6)

  @chex.all_variants
  @parameterized.parameters(
      {'batch_shape': (), 'param_shape': ()},
      {'batch_shape': (3,), 'param_shape': (3,)},
      {'batch_shape': (2, 3), 'param_shape': (3,)},
  )
  def test_identity_initialization(self, batch_shape, param_shape):
    bijector = LowerUpperTriangularAffine(
        matrix=jnp.eye(4),
        bias=jnp.zeros(param_shape + (4,)))
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    x = jax.random.normal(next(prng), batch_shape + (4,))

    # Forward methods.
    y, logdet = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y, x, 8e-3)
    np.testing.assert_array_equal(logdet, jnp.zeros(batch_shape))

    # Inverse methods.
    x_rec, logdet = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_array_equal(x_rec, y)
    np.testing.assert_array_equal(logdet, jnp.zeros(batch_shape))

  @chex.all_variants
  @parameterized.parameters(
      {'batch_shape': (), 'param_shape': ()},
      {'batch_shape': (3,), 'param_shape': (3,)},
      {'batch_shape': (2, 3), 'param_shape': (3,)}
  )
  def test_inverse_methods(self, batch_shape, param_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    matrix = jax.random.uniform(next(prng), param_shape + (4, 4)) + jnp.eye(4)
    bias = jax.random.normal(next(prng), param_shape + (4,))
    bijector = LowerUpperTriangularAffine(matrix, bias)
    x = jax.random.normal(next(prng), batch_shape + (4,))
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    x_rec, logdet_inv = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_rec, x, atol=9e-3)
    np.testing.assert_array_equal(logdet_fwd, -logdet_inv)

  @chex.all_variants
  def test_forward_jacobian_det(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    matrix = jax.random.uniform(next(prng), (4, 4)) + jnp.eye(4)
    bias = jax.random.normal(next(prng), (4,))
    bijector = LowerUpperTriangularAffine(matrix, bias)

    batched_x = jax.random.normal(next(prng), (10, 4))
    single_x = jax.random.normal(next(prng), (4,))
    batched_logdet = self.variant(bijector.forward_log_det_jacobian)(batched_x)

    jacobian_fn = jax.jacfwd(bijector.forward)
    logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_x))[1]
    for logdet in batched_logdet:
      np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-3)

  @chex.all_variants
  def test_inverse_jacobian_det(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    matrix = jax.random.uniform(next(prng), (4, 4)) + jnp.eye(4)
    bias = jax.random.normal(next(prng), (4,))
    bijector = LowerUpperTriangularAffine(matrix, bias)

    batched_y = jax.random.normal(next(prng), (10, 4))
    single_y = jax.random.normal(next(prng), (4,))
    batched_logdet = self.variant(bijector.inverse_log_det_jacobian)(batched_y)

    jacobian_fn = jax.jacfwd(bijector.inverse)
    logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_y))[1]
    for logdet in batched_logdet:
      np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-5)

  def test_raises_on_invalid_input_shape(self):
    bij = LowerUpperTriangularAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.assertRaises(ValueError):
        fn(jnp.array(0))

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = LowerUpperTriangularAffine(
        matrix=jnp.eye(4),
        bias=jnp.zeros((4,)))
    x = np.zeros((4,))
    f(x, bijector)

  def test_same_as_itself(self):
    bij = LowerUpperTriangularAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
    self.assertTrue(bij.same_as(bij))
    self.assertTrue(bij.same_as(copy.copy(bij)))

  def test_not_same_as_others(self):
    bij = LowerUpperTriangularAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
    other = LowerUpperTriangularAffine(matrix=jnp.eye(4), bias=jnp.ones((4,)))
    self.assertFalse(bij.same_as(other))
    self.assertFalse(bij.same_as(Tanh()))


if __name__ == '__main__':
  absltest.main()
