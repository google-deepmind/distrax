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
"""Logistic log densities stay finite throughout representable tails."""

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


class LogisticTailTest(parameterized.TestCase):

  @parameterized.product(dtype=(np.float32, np.float64), compiled=(False, True))
  def test_finite_tails_and_infinite_limits_match_scipy(self, dtype, compiled):
    previous = jax.config.x64_enabled
    self.addCleanup(jax.config.update, 'jax_enable_x64', previous)
    jax.config.update('jax_enable_x64', True)
    extreme = np.finfo(dtype).max * 0.75
    values = jnp.asarray(
        [-extreme, -100.0, -1.0, 0.0, 1.0, 100.0, extreme, -np.inf, np.inf],
        dtype=dtype,
    )
    distribution = distrax.Logistic(
        jnp.asarray(0.0, dtype), jnp.asarray(1.0, dtype)
    )
    fn = jax.jit(distribution.log_prob) if compiled else distribution.log_prob
    result = np.asarray(fn(values))
    expected = stats.logistic.logpdf(np.asarray(values))
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)
    self.assertTrue(np.isfinite(result[:7]).all())
    np.testing.assert_array_equal(result[-2:], [-np.inf, -np.inf])
    self.assertEqual(result.dtype, dtype)
    np.testing.assert_allclose(result[:7], result[:7][::-1], rtol=2e-6)

  def test_broadcast_parameters_and_sample_axes(self):
    locations = jnp.array([[0.0], [3.0]], dtype=jnp.float32)
    scales = jnp.array([[1.0, 2.0, 4.0]], dtype=jnp.float32)
    values = jnp.array([-2e38, -3.0, 0.0, 3.0, 2e38], dtype=jnp.float32)[
        :, None, None
    ]
    distribution = distrax.Logistic(locations, scales)
    expected = stats.logistic.logpdf(
        np.asarray(values, dtype=np.float64),
        np.asarray(locations),
        np.asarray(scales),
    )
    result = jax.jit(distribution.log_prob)(values)
    self.assertEqual(result.shape, (5, 2, 3))
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(
        distribution[1].log_prob(values[:, 0, :]), result[:, 1, :], rtol=2e-6
    )

  def test_value_and_derivatives_at_center_and_tails(self):
    dist = distrax.Logistic(0.0, 1.0)
    values = jnp.array(
        [-2e38, -100.0, -1.0, 0.0, 1.0, 100.0, 2e38], dtype=jnp.float32
    )
    value, gradient = jax.jit(jax.vmap(jax.value_and_grad(dist.log_prob)))(
        values
    )
    self.assertTrue(np.isfinite(value).all())
    np.testing.assert_allclose(
        gradient, -np.tanh(np.asarray(values) / 2.0), rtol=2e-6, atol=1e-7
    )
    center_second_derivative = jax.grad(jax.grad(dist.log_prob))(0.0)
    self.assertAlmostEqual(float(center_second_derivative), -0.5, places=6)
    loss, location_gradient = jax.jit(
        jax.value_and_grad(
            lambda loc: -distrax.Logistic(loc, 1.0).log_prob(jnp.float32(-2e38))
        )
    )(0.0)
    self.assertTrue(np.isfinite(loss))
    self.assertAlmostEqual(float(location_gradient), 1.0, places=6)

  def test_sampling_and_sample_log_prob_retain_the_same_draws(self):
    dist = distrax.Logistic(jnp.array([0.0, 2.0]), jnp.array([1.0, 3.0]))
    key = jax.random.key(29)
    samples = jax.jit(lambda k: dist.sample(seed=k, sample_shape=(4, 3)))(key)
    paired, log_probs = jax.jit(
        lambda k: dist.sample_and_log_prob(seed=k, sample_shape=(4, 3))
    )(key)
    np.testing.assert_array_equal(paired, samples)
    np.testing.assert_allclose(
        log_probs,
        stats.logistic.logpdf(
            np.asarray(samples), np.array([0.0, 2.0]), np.array([1.0, 3.0])
        ),
        rtol=3e-6,
        atol=1e-6,
    )


if __name__ == '__main__':
  absltest.main()
