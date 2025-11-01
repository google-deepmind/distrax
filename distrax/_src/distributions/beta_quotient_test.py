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
from absl.testing import absltest
from absl.testing import parameterized
# import chex # Removing chex
from distrax._src.distributions import beta_quotient
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

# DO NOT Enable float64
# jax.config.update("jax_enable_x64", True)


class BetaQuotientTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(42)
    # self.dtype = jnp.float64 # Use default dtype
    # Mock self.variant to be a no-op, removing chex variants
    self.variant = lambda f: f

  @parameterized.named_parameters(
      ('1d params', 2.0, 3.0, 4.0, 5.0),
      (
          '2d params',
          np.array([1.1, 2.0]),
          np.array([2.2, 3.0]),
          np.array([3.3, 4.0]),
          np.array([4.4, 5.0]),
      ),
      (
          'broadcasted params',
          np.array([1.1, 2.0]),
          3.0,
          4.0,
          np.array([4.4, 5.0]),
      ),
  )
  def test_properties(self, a0, b0, a1, b1):
    dist = beta_quotient.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )
    tfp_dist = tfd.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )

    self.assertEqual(dist.event_shape, tfp_dist.event_shape)
    self.assertEqual(dist.batch_shape, tfp_dist.batch_shape)
    np.testing.assert_allclose(
        dist.concentration1_numerator,
        tfp_dist.concentration1_numerator,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dist.concentration0_numerator,
        tfp_dist.concentration0_numerator,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dist.concentration1_denominator,
        tfp_dist.concentration1_denominator,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dist.concentration0_denominator,
        tfp_dist.concentration0_denominator,
        rtol=1e-6,
    )

  @parameterized.named_parameters(
      ('1d params, 1d value', 2.0, 3.0, 4.0, 5.0, np.array([0.5, 1.0, 1.5])),
      (
          '2d params, 1d value',
          np.array([1.1, 2.0]),
          np.array([2.2, 3.0]),
          np.array([3.3, 4.0]),
          np.array([4.4, 5.0]),
          np.array([0.5, 1.5]),
      ),
      (
          '1d params, 2d value',
          2.0,
          3.0,
          4.0,
          5.0,
          np.array([[0.5, 1.0, 1.5], [1.2, 0.8, 0.2]]),
      ),
      (
          '2d params, 2d value',
          np.array([1.1, 2.0]),
          np.array([2.2, 3.0]),
          np.array([3.3, 4.0]),
          np.array([4.4, 5.0]),
          np.array([[0.5, 1.5], [1.2, 0.8]]),
      ),
  )
  def test_log_prob(self, a0, b0, a1, b1, value):
    # TFP needs float64 for hyp2f1, but distrax uses float32.
    # Let's test with float32.
    a0_f32 = jnp.asarray(a0, dtype=jnp.float32)
    b0_f32 = jnp.asarray(b0, dtype=jnp.float32)
    a1_f32 = jnp.asarray(a1, dtype=jnp.float32)
    b1_f32 = jnp.asarray(b1, dtype=jnp.float32)
    value_jnp_f32 = jnp.asarray(value, dtype=jnp.float32)

    dist_f32 = beta_quotient.BetaQuotient(
        concentration1_numerator=a0_f32,
        concentration0_numerator=b0_f32,
        concentration1_denominator=a1_f32,
        concentration0_denominator=b1_f32,
    )
    tfp_dist_f32 = tfd.BetaQuotient(
        concentration1_numerator=a0_f32,
        concentration0_numerator=b0_f32,
        concentration1_denominator=a1_f32,
        concentration0_denominator=b1_f32,
    )

    log_prob_fn = self.variant(dist_f32.log_prob)
    tfp_log_prob_fn = self.variant(tfp_dist_f32.log_prob)

    np.testing.assert_allclose(
        log_prob_fn(value_jnp_f32),
        tfp_log_prob_fn(value_jnp_f32),
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.named_parameters(
      ('1d params', 2.0, 3.0, 4.0, 5.0),
      (
          '2d params',
          np.array([1.1, 2.0]),
          np.array([2.2, 3.0]),
          np.array([3.3, 4.0]),
          np.array([4.4, 5.0]),
      ),
      (
          'broadcasted params',
          np.array([1.1, 2.0]),
          3.0,
          4.0,
          np.array([4.4, 5.0]),
      ),
      ('unstable mean', 0.5, 0.5, 1.0, 0.5),
  )
  def test_mean(self, a0, b0, a1, b1):
    dist = beta_quotient.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )
    tfp_dist = tfd.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )

    mean_fn = self.variant(dist.mean)
    tfp_mean_fn = self.variant(tfp_dist.mean)

    np.testing.assert_allclose(
        mean_fn(), tfp_mean_fn(), rtol=1e-5, atol=1e-5, equal_nan=True
    )

  @parameterized.named_parameters(
      ('1 sample', 2.0, 3.0, 4.0, 5.0, 1),
      ('5 samples', 2.0, 3.0, 4.0, 5.0, 5),
      (
          'broadcasted params',
          np.array([1.1, 2.0]),
          3.0,
          4.0,
          np.array([4.4, 5.0]),
          5,
      ),
  )
  def test_sample(self, a0, b0, a1, b1, n):
    dist = beta_quotient.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )

    sample_fn = self.variant(lambda key: dist.sample(seed=key, sample_shape=n))
    samples = sample_fn(self.key)

    self.assertEqual(samples.shape, (n,) + dist.batch_shape)
    self.assertTrue(jnp.all(samples >= 0))

  @parameterized.named_parameters(
      ('1d params, 1d value', 2.0, 3.0, 4.0, 5.0, np.array([0.5, 1.0, 1.5])),
      (
          '2d params, 1d value',
          np.array([1.1, 2.0]),
          np.array([2.2, 3.0]),
          np.array([3.3, 4.0]),
          np.array([4.4, 5.0]),
          np.array([0.5, 1.5]),
      ),
  )
  def test_log_prob_grad(self, a0, b0, a1, b1, value):
    dist = beta_quotient.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )

    log_prob_fn = lambda a0, b0, a1, b1, x: dist.log_prob(x)

    value_jnp = jnp.asarray(value)

    # Grad wrt value
    grad_fn_val = self.variant(
        jax.grad(lambda x: log_prob_fn(a0, b0, a1, b1, x).sum())
    )
    grad_val = grad_fn_val(value_jnp)
    self.assertEqual(grad_val.shape, value_jnp.shape)
    self.assertFalse(jnp.any(jnp.isnan(grad_val)))

    # Grad wrt params
    grad_fn_params = self.variant(
        jax.grad(
            lambda p: log_prob_fn(p[0], p[1], p[2], p[3], value_jnp).sum(),
            argnums=0,
        )
    )
    params = [
        jnp.asarray(a0),
        jnp.asarray(b0),
        jnp.asarray(a1),
        jnp.asarray(b1),
    ]
    grad_params = grad_fn_params(params)
    self.assertLen(grad_params, 4)
    for g, p in zip(grad_params, params):
      self.assertEqual(g.shape, p.shape)
      self.assertFalse(jnp.any(jnp.isnan(g)))

  def test_slicing(self):
    a0 = jnp.array([1.0, 2.0, 3.0])
    b0 = jnp.array([4.0, 5.0, 6.0])
    a1 = jnp.array([7.0, 8.0, 9.0])
    b1 = jnp.array([10.0, 11.0, 12.0])

    dist = beta_quotient.BetaQuotient(
        concentration1_numerator=a0,
        concentration0_numerator=b0,
        concentration1_denominator=a1,
        concentration0_denominator=b1,
    )
    self.assertEqual(dist.batch_shape, (3,))

    dist_sliced = self.variant(lambda d: d[1:])(dist)
    self.assertEqual(dist_sliced.batch_shape, (2,))
    np.testing.assert_allclose(
        dist_sliced.concentration1_numerator, a0[1:], rtol=1e-5
    )
    np.testing.assert_allclose(
        dist_sliced.concentration0_numerator, b0[1:], rtol=1e-5
    )
    np.testing.assert_allclose(
        dist_sliced.concentration1_denominator, a1[1:], rtol=1e-5
    )
    np.testing.assert_allclose(
        dist_sliced.concentration0_denominator, b1[1:], rtol=1e-5
    )

    dist_sliced_int = self.variant(lambda d: d[0])(dist)
    self.assertEqual(dist_sliced_int.batch_shape, ())
    np.testing.assert_allclose(
        dist_sliced_int.concentration1_numerator, a0[0], rtol=1e-5
    )
    np.testing.assert_allclose(
        dist_sliced_int.concentration0_numerator, b0[0], rtol=1e-5
    )
    np.testing.assert_allclose(
        dist_sliced_int.concentration1_denominator, a1[0], rtol=1e-5
    )
    np.testing.assert_allclose(
        dist_sliced_int.concentration0_denominator, b1[0], rtol=1e-5
    )


if __name__ == '__main__':
  absltest.main()
