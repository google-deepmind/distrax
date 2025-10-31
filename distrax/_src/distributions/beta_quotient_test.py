# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
from distrax._src.distributions import beta_quotient
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class BetaQuotientTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._dist = beta_quotient.BetaQuotient
    self._tfp_dist = tfd.BetaQuotient

  @parameterized.named_parameters(
      ('1d', (2.0, 3.0, 4.0, 5.0)),
      ('2d', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
              jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0]))),
  )
  def test_params(self, params):
    a0, b0, a1, b1 = params
    dist = self._dist(a0, b0, a1, b1)
    tfp_dist = self._tfp_dist(a0, b0, a1, b1)

    np.testing.assert_allclose(
        dist.concentration1_numerator, tfp_dist.concentration1_numerator)
    np.testing.assert_allclose(
        dist.concentration0_numerator, tfp_dist.concentration0_numerator)
    np.testing.assert_allclose(
        dist.concentration1_denominator, tfp_dist.concentration1_denominator)
    np.testing.assert_allclose(
        dist.concentration0_denominator, tfp_dist.concentration0_denominator)

  @parameterized.named_parameters(
      ('1d', (2.0, 3.0, 4.0, 5.0), ()),
      ('2d', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
              jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0]), (2,))),
  )
  def test_shapes(self, params, batch_shape):
    a0, b0, a1, b1 = params
    dist = self._dist(a0, b0, a1, b1)
    tfp_dist = self._tfp_dist(a0, b0, a1, b1)

    self.assertEqual(dist.event_shape, tfp_dist.event_shape)
    self.assertEqual(dist.batch_shape, tfp_dist.batch_shape)
    self.assertEqual(dist.batch_shape, batch_shape)

  @parameterized.named_parameters(
      ('1d', (2.0, 3.0, 4.0, 5.0)),
      ('2d', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
              jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0]))),
  )
  def test_mean(self, params):
    a0, b0, a1, b1 = params
    dist = self._dist(a0, b0, a1, b1)
    tfp_dist = self._tfp_dist(a0, b0, a1, b1)

    np.testing.assert_allclose(dist.mean(), tfp_dist.mean(), atol=1e-5)

  @parameterized.named_parameters(
      ('1d_a1_fail', (2.0, 3.0, 0.5, 5.0)),
      ('2d_a1_fail', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
                      jnp.array([0.5, 6.0]), jnp.array([7.0, 8.0]))),
  )
  def test_mean_nan(self, params):
    a0, b0, a1, b1 = params
    dist = self._dist(a0, b0, a1, b1)
    tfp_dist = self._tfp_dist(a0, b0, a1, b1)

    np.testing.assert_allclose(dist.mean(), tfp_dist.mean())

  @parameterized.named_parameters(
      ('1d', (2.0, 3.0, 4.0, 5.0), 0.5),
      ('2d', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
              jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0])),
       jnp.array([0.2, 0.7])),
      ('x_gt_1', (2.0, 3.0, 4.0, 5.0), 1.5),
      ('x_gt_1_2d', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
                     jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0])),
       jnp.array([1.2, 2.7])),
  )
  def test_log_prob(self, params, x):
    a0, b0, a1, b1 = params
    dist = self._dist(a0, b0, a1, b1)
    tfp_dist = self._tfp_dist(a0, b0, a1, b1)

    # TFP log_prob has some numerical stability issues near z=1,
    # which my implementation tries to mitigate by clipping.
    # The TFP implementation uses hyp2f1_small_argument which also does
    # transformations.
    # Let's compare with a small tolerance.
    np.testing.assert_allclose(
        dist.log_prob(x), tfp_dist.log_prob(x), atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      ('1d', (2.0, 3.0, 4.0, 5.0)),
      ('2d', (jnp.array([1.1, 2.0]), jnp.array([3.0, 4.0]),
              jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0]))),
  )
  def test_sample(self, params):
    a0, b0, a1, b1 = params
    dist = self._dist(a0, b0, a1, b1)
    key = jax.random.PRNGKey(42)
    samples = dist.sample(seed=key, sample_shape=(1000,))
    self.assertEqual(samples.shape, (1000,) + dist.batch_shape)

    # Check that samples are positive
    self.assertTrue(jnp.all(samples > 0.))

  def test_jittable(self):
    a0, b0, a1, b1 = 2.0, 3.0, 4.0, 5.0
    dist = self._dist(a0, b0, a1, b1)

    # Check that jitting doesn't crash
    jax.jit(dist.mean)()
    jax.jit(dist.log_prob)(value=0.5)
    jax.jit(dist.sample)(seed=jax.random.PRNGKey(42), sample_shape=())

if __name__ == '__main__':
  absltest.main()

