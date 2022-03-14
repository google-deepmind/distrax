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
"""Tests for `shift.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors.shift import Shift
from distrax._src.bijectors.tanh import Tanh
import jax
import jax.numpy as jnp
import numpy as np


class ShiftTest(parameterized.TestCase):

  def test_jacobian_is_constant_property(self):
    bijector = Shift(jnp.ones((4,)))
    self.assertTrue(bijector.is_constant_jacobian)
    self.assertTrue(bijector.is_constant_log_det)

  def test_properties(self):
    bijector = Shift(jnp.array([1., 2., 3.]))
    np.testing.assert_array_equal(bijector.shift, np.array([1., 2., 3.]))

  @chex.all_variants
  @parameterized.parameters(
      {'batch_shape': (), 'param_shape': ()},
      {'batch_shape': (3,), 'param_shape': ()},
      {'batch_shape': (), 'param_shape': (3,)},
      {'batch_shape': (2, 3), 'param_shape': (2, 3)},
  )
  def test_forward_methods(self, batch_shape, param_shape):
    bijector = Shift(jnp.ones(param_shape))
    prng = jax.random.PRNGKey(42)
    x = jax.random.normal(prng, batch_shape)
    output_shape = jnp.broadcast_shapes(batch_shape, param_shape)
    y1 = self.variant(bijector.forward)(x)
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    self.assertEqual(y1.shape, output_shape)
    self.assertEqual(y2.shape, output_shape)
    self.assertEqual(logdet1.shape, output_shape)
    self.assertEqual(logdet2.shape, output_shape)
    np.testing.assert_allclose(y1, x + 1., 1e-6)
    np.testing.assert_allclose(y2, x + 1., 1e-6)
    np.testing.assert_allclose(logdet1, 0., 1e-6)
    np.testing.assert_allclose(logdet2, 0., 1e-6)

  @chex.all_variants
  @parameterized.parameters(
      {'batch_shape': (), 'param_shape': ()},
      {'batch_shape': (3,), 'param_shape': ()},
      {'batch_shape': (), 'param_shape': (3,)},
      {'batch_shape': (2, 3), 'param_shape': (2, 3)},
  )
  def test_inverse_methods(self, batch_shape, param_shape):
    bijector = Shift(jnp.ones(param_shape))
    prng = jax.random.PRNGKey(42)
    y = jax.random.normal(prng, batch_shape)
    output_shape = jnp.broadcast_shapes(batch_shape, param_shape)
    x1 = self.variant(bijector.inverse)(y)
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    self.assertEqual(x1.shape, output_shape)
    self.assertEqual(x2.shape, output_shape)
    self.assertEqual(logdet1.shape, output_shape)
    self.assertEqual(logdet2.shape, output_shape)
    np.testing.assert_allclose(x1, y - 1., 1e-6)
    np.testing.assert_allclose(x2, y - 1., 1e-6)
    np.testing.assert_allclose(logdet1, 0., 1e-6)
    np.testing.assert_allclose(logdet2, 0., 1e-6)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bij = Shift(jnp.ones((4,)))
    x = np.zeros((4,))
    f(x, bij)

  def test_same_as_itself(self):
    bij = Shift(jnp.ones((4,)))
    self.assertTrue(bij.same_as(bij))

  def test_not_same_as_others(self):
    bij = Shift(jnp.ones((4,)))
    other = Shift(jnp.zeros((4,)))
    self.assertFalse(bij.same_as(other))
    self.assertFalse(bij.same_as(Tanh()))


if __name__ == '__main__':
  absltest.main()
