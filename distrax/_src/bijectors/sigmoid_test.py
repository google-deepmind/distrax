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
"""Tests for `sigmoid.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import sigmoid
from distrax._src.bijectors import tanh
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors

RTOL = 1e-5


class SigmoidTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  def test_properties(self):
    bijector = sigmoid.Sigmoid()
    self.assertEqual(bijector.event_ndims_in, 0)
    self.assertEqual(bijector.event_ndims_out, 0)
    self.assertFalse(bijector.is_constant_jacobian)
    self.assertFalse(bijector.is_constant_log_det)

  @chex.all_variants
  @parameterized.parameters(
      {'x_shape': (2,)},
      {'x_shape': (2, 3)},
      {'x_shape': (2, 3, 4)})
  def test_forward_shapes(self, x_shape):
    x = jnp.zeros(x_shape)
    bijector = sigmoid.Sigmoid()
    y1 = self.variant(bijector.forward)(x)
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    self.assertEqual(y1.shape, x_shape)
    self.assertEqual(y2.shape, x_shape)
    self.assertEqual(logdet1.shape, x_shape)
    self.assertEqual(logdet2.shape, x_shape)

  @chex.all_variants
  @parameterized.parameters(
      {'y_shape': (2,)},
      {'y_shape': (2, 3)},
      {'y_shape': (2, 3, 4)})
  def test_inverse_shapes(self, y_shape):
    y = jnp.zeros(y_shape)
    bijector = sigmoid.Sigmoid()
    x1 = self.variant(bijector.inverse)(y)
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    self.assertEqual(x1.shape, y_shape)
    self.assertEqual(x2.shape, y_shape)
    self.assertEqual(logdet1.shape, y_shape)
    self.assertEqual(logdet2.shape, y_shape)

  @chex.all_variants
  def test_forward(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    y = self.variant(bijector.forward)(x)
    np.testing.assert_allclose(y, jax.nn.sigmoid(x), rtol=RTOL)

  @chex.all_variants
  def test_forward_log_det_jacobian(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    fwd_logdet = self.variant(bijector.forward_log_det_jacobian)(x)
    actual = jnp.log(jax.vmap(jax.grad(bijector.forward))(x))
    np.testing.assert_allclose(fwd_logdet, actual, rtol=1e-3)

  @chex.all_variants
  def test_forward_and_log_det(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    y1 = self.variant(bijector.forward)(x)
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y1, y2, rtol=RTOL)
    np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

  @chex.all_variants
  def test_inverse(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    y = self.variant(bijector.forward)(x)
    x_rec = self.variant(bijector.inverse)(y)
    np.testing.assert_allclose(x_rec, x, rtol=1e-3)

  @chex.all_variants
  def test_inverse_log_det_jacobian(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    y = self.variant(bijector.forward)(x)
    fwd_logdet = self.variant(bijector.forward_log_det_jacobian)(x)
    inv_logdet = self.variant(bijector.inverse_log_det_jacobian)(y)
    np.testing.assert_allclose(inv_logdet, -fwd_logdet, rtol=1e-4)

  @chex.all_variants
  def test_inverse_and_log_det(self):
    y = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    x1 = self.variant(bijector.inverse)(y)
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x1, x2, rtol=RTOL)
    np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

  @chex.all_variants
  def test_stability(self):
    bijector = sigmoid.Sigmoid()
    tfp_bijector = tfb.Sigmoid()

    x = np.array([-10.0, -3.3, 0.0, 3.3, 10.0], dtype=np.float32)
    fldj = tfp_bijector.forward_log_det_jacobian(x, event_ndims=0)
    fldj_ = self.variant(bijector.forward_log_det_jacobian)(x)
    np.testing.assert_allclose(fldj_, fldj, rtol=RTOL)

    y = bijector.forward(x)
    ildj = tfp_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    ildj_ = self.variant(bijector.inverse_log_det_jacobian)(y)
    np.testing.assert_allclose(ildj_, ildj, rtol=RTOL)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = sigmoid.Sigmoid()
    x = np.zeros(())
    f(x, bijector)

  def test_same_as(self):
    bijector = sigmoid.Sigmoid()
    self.assertTrue(bijector.same_as(bijector))
    self.assertTrue(bijector.same_as(sigmoid.Sigmoid()))
    self.assertFalse(bijector.same_as(tanh.Tanh()))


if __name__ == '__main__':
  absltest.main()
