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
"""Tests for `tanh.py`."""

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


class TanhTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  def test_properties(self):
    bijector = tanh.Tanh()
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
    bijector = tanh.Tanh()
    y1 = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)  # pyrefly: ignore[missing-attribute]
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)  # pyrefly: ignore[missing-attribute]
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
    bijector = tanh.Tanh()
    x1 = self.variant(bijector.inverse)(y)  # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)  # pyrefly: ignore[missing-attribute]
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(x1.shape, y_shape)
    self.assertEqual(x2.shape, y_shape)
    self.assertEqual(logdet1.shape, y_shape)
    self.assertEqual(logdet2.shape, y_shape)

  @chex.all_variants
  def test_forward(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = tanh.Tanh()
    y = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(y, jnp.tanh(x), rtol=RTOL)

  @chex.all_variants
  def test_forward_log_det_jacobian(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = tanh.Tanh()
    fwd_logdet = self.variant(bijector.forward_log_det_jacobian)(x)  # pyrefly: ignore[missing-attribute]
    actual = jnp.log(jax.vmap(jax.grad(bijector.forward))(x))
    np.testing.assert_allclose(fwd_logdet, actual, rtol=1e-2)

  @chex.all_variants
  def test_forward_and_log_det(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = tanh.Tanh()
    y1 = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)  # pyrefly: ignore[missing-attribute]
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(y1, y2, rtol=RTOL)
    np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

  @chex.all_variants
  def test_inverse(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = tanh.Tanh()
    y = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    x_rec = self.variant(bijector.inverse)(y)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(x_rec, x, rtol=1e-3)

  @chex.all_variants
  def test_inverse_log_det_jacobian(self):
    x = jax.random.normal(self.seed, (100,))
    bijector = tanh.Tanh()
    y = self.variant(bijector.forward)(x)  # pyrefly: ignore[missing-attribute]
    fwd_logdet = self.variant(bijector.forward_log_det_jacobian)(x)  # pyrefly: ignore[missing-attribute]
    inv_logdet = self.variant(bijector.inverse_log_det_jacobian)(y)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(inv_logdet, -fwd_logdet, rtol=1e-3)

  @chex.all_variants
  def test_inverse_and_log_det(self):
    y = jax.random.normal(self.seed, (100,))
    bijector = tanh.Tanh()
    x1 = self.variant(bijector.inverse)(y)  # pyrefly: ignore[missing-attribute]
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)  # pyrefly: ignore[missing-attribute]
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(x1, x2, rtol=RTOL)
    np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

  @chex.all_variants
  def test_stability(self):
    bijector = tanh.Tanh()
    tfp_bijector = tfb.Tanh()

    x = np.array([-10.0, -3.3, 0.0, 3.3, 10.0], dtype=np.float32)
    fldj = tfp_bijector.forward_log_det_jacobian(x, event_ndims=0)
    fldj_ = self.variant(bijector.forward_log_det_jacobian)(x)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(fldj_, fldj, rtol=RTOL)

    y = bijector.forward(x)  # pytype: disable=wrong-arg-types  # jax-ndarray
    # For the inverse log-det, distrax clips boundary values to prevent NaN
    # (unlike TFP which returns NaN for tanh(±10) = ±1 in float32).
    # We verify finiteness rather than matching TFP's NaN for those entries.
    # Interior values (|x| < 10) still agree with TFP.
    ildj_ = self.variant(bijector.inverse_log_det_jacobian)(y)  # pyrefly: ignore[missing-attribute]
    self.assertFalse(np.any(np.isnan(ildj_)),
                     'inverse_log_det_jacobian should be finite, got NaN')
    interior = np.array([1, 2, 3], dtype=int)  # indices for x in {-3.3,0,3.3}
    ildj_tfp = tfp_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    np.testing.assert_allclose(ildj_[interior], ildj_tfp[interior], rtol=RTOL)

  @chex.all_variants
  @parameterized.named_parameters(
      ('int16', np.array([0, 0], dtype=np.int16)),
      ('int32', np.array([0, 0], dtype=np.int32)),
      ('int64', np.array([0, 0], dtype=np.int64)),
  )
  def test_integer_inputs(self, inputs):
    bijector = tanh.Tanh()
    output, log_det = self.variant(bijector.forward_and_log_det)(inputs)  # pyrefly: ignore[missing-attribute]

    expected_out = jnp.tanh(inputs).astype(jnp.float32)
    expected_log_det = jnp.zeros_like(inputs, dtype=jnp.float32)

    np.testing.assert_array_equal(output, expected_out)
    np.testing.assert_array_equal(log_det, expected_log_det)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = tanh.Tanh()
    x = np.zeros(())
    f(x, bijector)

  def test_same_as(self):
    bijector = tanh.Tanh()
    self.assertTrue(bijector.same_as(bijector))
    self.assertTrue(bijector.same_as(tanh.Tanh()))
    self.assertFalse(bijector.same_as(sigmoid.Sigmoid()))

  def test_inverse_clips_boundary_values_to_prevent_nan(self):
    """Regression test for https://github.com/google-deepmind/distrax/issues/216.

    In float32, sampling from a Tanh-transformed distribution can yield values
    numerically equal to ±1 due to limited precision.  arctanh(±1) = ±∞,
    which causes NaN in log_prob.  The fix clips y to (-1+eps, 1-eps).
    """
    bijector = tanh.Tanh()
    # Exact boundary values that would produce NaN without clipping.
    y_boundary = jnp.array([-1.0, 1.0], dtype=jnp.float32)
    x, log_det = bijector.inverse_and_log_det(y_boundary)
    self.assertFalse(jnp.any(jnp.isnan(x)), 'x should be finite, got NaN')
    self.assertFalse(jnp.any(jnp.isnan(log_det)),
                     'log_det should be finite, got NaN')
    self.assertFalse(jnp.any(jnp.isinf(x)), 'x should be finite, got Inf')
    self.assertFalse(jnp.any(jnp.isinf(log_det)),
                     'log_det should be finite, got Inf')

  def test_log_prob_finite_at_float32_boundary_samples(self):
    """log_prob must be finite for samples that saturate float32 tanh."""
    import distrax
    import jax.random as jr
    # Build a Tanh-wrapped normal that is likely to produce boundary samples.
    dist = distrax.Transformed(
        distribution=distrax.MultivariateNormalDiag(
            loc=jnp.zeros(4, dtype=jnp.float32),
            scale_diag=jnp.ones(4, dtype=jnp.float32) * 10.0),  # wide → ±1
        bijector=distrax.Block(distrax.Tanh(), ndims=1))
    key = jr.PRNGKey(0)
    samples = dist.sample(seed=key, sample_shape=(16,))
    log_probs = dist.log_prob(samples)
    self.assertFalse(jnp.any(jnp.isnan(log_probs)),
                     'log_prob returned NaN for float32 boundary samples')


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
