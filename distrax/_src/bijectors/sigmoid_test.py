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
import distrax
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
    y1 = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    # pyrefly: ignore[missing-attribute]
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
    x1 = self.variant(bijector.inverse)(y)  # pyrefly: ignore[missing-attribute]
    # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    # pyrefly: ignore[missing-attribute]
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    self.assertEqual(x1.shape, y_shape)
    self.assertEqual(x2.shape, y_shape)
    self.assertEqual(logdet1.shape, y_shape)
    self.assertEqual(logdet2.shape, y_shape)

  @chex.all_variants
  def test_forward(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    # pyrefly: ignore[missing-attribute]
    y = self.variant(bijector.forward)(x)
    np.testing.assert_allclose(y, jax.nn.sigmoid(x), rtol=RTOL)

  @chex.all_variants
  def test_forward_log_det_jacobian(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    # pyrefly: ignore[missing-attribute]
    fwd_logdet = self.variant(bijector.forward_log_det_jacobian)(x)
    actual = jnp.log(jax.vmap(jax.grad(bijector.forward))(x))
    np.testing.assert_allclose(fwd_logdet, actual, rtol=1e-3)

  @chex.all_variants
  def test_forward_and_log_det(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    y1 = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    # pyrefly: ignore[missing-attribute]
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y1, y2, rtol=RTOL)
    np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

  @chex.all_variants
  def test_inverse(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    # pyrefly: ignore[missing-attribute]
    y = self.variant(bijector.forward)(x)
    # pyrefly: ignore[missing-attribute]
    x_rec = self.variant(bijector.inverse)(y)
    np.testing.assert_allclose(x_rec, x, rtol=1e-3)

  @chex.all_variants
  def test_inverse_log_det_jacobian(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    y = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    # pyrefly: ignore[missing-attribute]
    fwd_logdet = self.variant(bijector.forward_log_det_jacobian)(x)
    # pyrefly: ignore[missing-attribute]
    inv_logdet = self.variant(bijector.inverse_log_det_jacobian)(y)
    np.testing.assert_allclose(inv_logdet, -fwd_logdet, rtol=1e-4)

  @chex.all_variants
  def test_inverse_and_log_det(self):
    y = jax.random.normal(self.seed, (100,))
    bijector = sigmoid.Sigmoid()
    x1 = self.variant(bijector.inverse)(y)  # pyrefly: ignore[missing-attribute]
    # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    # pyrefly: ignore[missing-attribute]
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x1, x2, rtol=RTOL)
    np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

  @chex.all_variants
  def test_stability(self):
    bijector = sigmoid.Sigmoid()
    tfp_bijector = tfb.Sigmoid()

    x = np.array([-10.0, -3.3, 0.0, 3.3, 10.0], dtype=np.float32)
    fldj = tfp_bijector.forward_log_det_jacobian(x, event_ndims=0)
    # pyrefly: ignore[missing-attribute]
    fldj_ = self.variant(bijector.forward_log_det_jacobian)(x)
    np.testing.assert_allclose(fldj_, fldj, rtol=RTOL)

    y = bijector.forward(x)
    ildj = tfp_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    # pyrefly: ignore[missing-attribute]
    ildj_ = self.variant(bijector.inverse_log_det_jacobian)(y)
    np.testing.assert_allclose(ildj_, ildj, rtol=RTOL)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = sigmoid.Sigmoid()
    x = np.zeros(())
    f(x, bijector)

  @parameterized.product(
      use_jit=[False, True], dtype=[jnp.float32, jnp.float64])
  def test_forward_tail_gradients(self, use_jit, dtype):
    with jax.enable_x64(dtype == jnp.float64):
      x = jnp.array([-1000., -100., -80., -10., -9., 0., 10., 100., 1000.],
                    dtype=dtype)
      bijector = sigmoid.Sigmoid()
      gradient = jax.vmap(jax.grad(bijector.forward))
      if use_jit:
        gradient = jax.jit(gradient)
      expected = jnp.where(
          x < -9, jnp.exp(jnp.minimum(x, 0)),
          jax.vmap(jax.grad(jax.nn.sigmoid))(x))
      np.testing.assert_allclose(gradient(x), expected, rtol=RTOL, atol=0.)

  @parameterized.product(
      use_jit=[False, True], dtype=[jnp.float32, jnp.float64])
  def test_log_det_tail_gradients(self, use_jit, dtype):
    with jax.enable_x64(dtype == jnp.float64):
      x = jnp.array([-1000., -100., -80., -10., -9., 0., 9., 10., 80.,
                     100., 1000.], dtype=dtype)
      bijector = sigmoid.Sigmoid()
      gradient = jax.vmap(jax.grad(bijector.forward_log_det_jacobian))
      if use_jit:
        gradient = jax.jit(gradient)
      expected = -np.tanh(np.asarray(x) / 2)
      np.testing.assert_allclose(gradient(x), expected, rtol=RTOL, atol=1e-7)

  @parameterized.product(
      use_jit=[False, True], dtype=[jnp.float32, jnp.float64])
  def test_transformed_sample_log_prob_tail_gradients(self, use_jit, dtype):
    with jax.enable_x64(dtype == jnp.float64):
      loc = jnp.array([-1000., 1000.], dtype=dtype)

      def sample_log_prob(loc):
        distribution = distrax.Transformed(
            distrax.Normal(loc, jnp.ones_like(loc)), sigmoid.Sigmoid())
        samples, log_prob = distribution.sample_and_log_prob(
            seed=self.seed, sample_shape=(2, 3))
        return jnp.sum(jnp.mean(log_prob, axis=(0, 1))), samples

      value_and_grad = jax.value_and_grad(sample_log_prob, has_aux=True)
      if use_jit:
        value_and_grad = jax.jit(value_and_grad)
      (value, samples), gradient = value_and_grad(loc)
      self.assertEqual(samples.shape, (2, 3, 2))
      self.assertTrue(np.all(np.isfinite(samples)))
      self.assertTrue(np.isfinite(value))
      np.testing.assert_allclose(gradient, [-1., 1.], rtol=RTOL)

  def test_same_as(self):
    bijector = sigmoid.Sigmoid()
    self.assertTrue(bijector.same_as(bijector))
    self.assertTrue(bijector.same_as(sigmoid.Sigmoid()))
    self.assertFalse(bijector.same_as(tanh.Tanh()))


if __name__ == '__main__':
  absltest.main()
