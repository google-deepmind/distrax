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
"""Tests for the Kumaraswamy distribution."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions.kumaraswamy import Kumaraswamy
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

RTOL = 1e-5


class KumaraswamyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._a = np.array([0.5, 1.0, 2.0, 3.5], dtype=np.float32)
    self._b = np.array([2.0, 1.0, 3.0, 0.5], dtype=np.float32)

  # ── construction ─────────────────────────────────────────────────────────────

  def test_properties(self):
    dist = Kumaraswamy(self._a, self._b)
    self.assertEqual(dist.batch_shape, (4,))
    self.assertEqual(dist.event_shape, ())
    np.testing.assert_array_equal(dist.concentration0, self._a)
    np.testing.assert_array_equal(dist.concentration1, self._b)

  # ── log_prob ─────────────────────────────────────────────────────────────────

  @chex.all_variants()
  def test_log_prob_matches_tfp(self):
    dist = Kumaraswamy(self._a, self._b)
    tfp_dist = tfd.Kumaraswamy(self._a, self._b)
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    for x in xs:
      with self.subTest(x=x):
        our_lp = self.variant(dist.log_prob)(x)
        tfp_lp = tfp_dist.log_prob(x)
        np.testing.assert_allclose(our_lp, tfp_lp, rtol=1e-4)

  # ── log_cdf ───────────────────────────────────────────────────────────────────

  @chex.all_variants()
  def test_log_cdf_matches_tfp(self):
    dist = Kumaraswamy(2.0, 3.0)
    tfp_dist = tfd.Kumaraswamy(2.0, 3.0)
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    np.testing.assert_allclose(
        self.variant(dist.log_cdf)(xs),
        tfp_dist.log_cdf(xs), rtol=1e-4)

  # ── mean ─────────────────────────────────────────────────────────────────────

  def test_mean_matches_tfp(self):
    dist = Kumaraswamy(self._a, self._b)
    tfp_dist = tfd.Kumaraswamy(self._a, self._b)
    np.testing.assert_allclose(dist.mean(), tfp_dist.mean(), rtol=1e-4)

  # ── variance ─────────────────────────────────────────────────────────────────

  def test_variance_matches_tfp(self):
    dist = Kumaraswamy(self._a, self._b)
    tfp_dist = tfd.Kumaraswamy(self._a, self._b)
    np.testing.assert_allclose(dist.variance(), tfp_dist.variance(), rtol=1e-4)

  # ── mode ─────────────────────────────────────────────────────────────────────

  def test_mode_for_valid_interior(self):
    """Mode = ((a-1)/(ab-1))^(1/a) for a,b >= 1 and ab > 1."""
    a, b = 2.0, 3.0
    dist = Kumaraswamy(a, b)
    expected = float(((a - 1) / (a * b - 1)) ** (1 / a))
    np.testing.assert_allclose(float(dist.mode()), expected, rtol=RTOL)

  def test_mode_at_boundaries(self):
    """mode=0 when a<1 (density diverges at 0), mode=1 when b<1."""
    np.testing.assert_array_equal(Kumaraswamy(0.5, 2.0).mode(), 0.0)
    np.testing.assert_array_equal(Kumaraswamy(2.0, 0.5).mode(), 1.0)

  # ── entropy ───────────────────────────────────────────────────────────────────

  def test_entropy_matches_tfp(self):
    dist = Kumaraswamy(self._a, self._b)
    tfp_dist = tfd.Kumaraswamy(self._a, self._b)
    np.testing.assert_allclose(dist.entropy(), tfp_dist.entropy(), rtol=1e-4)

  def test_entropy_uniform_is_zero(self):
    """Kumaraswamy(1,1) = Uniform(0,1), entropy = 0."""
    dist = Kumaraswamy(1.0, 1.0)
    np.testing.assert_allclose(float(dist.entropy()), 0.0, atol=1e-6)

  # ── sampling ─────────────────────────────────────────────────────────────────

  def test_sample_shape(self):
    dist = Kumaraswamy(self._a, self._b)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(50,))
    self.assertEqual(samples.shape, (50, 4))

  def test_sample_range(self):
    dist = Kumaraswamy(2.0, 3.0)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(5000,))
    self.assertTrue(jnp.all((samples > 0) & (samples < 1)))

  def test_sample_moments(self):
    """Empirical mean should be close to analytical mean."""
    a, b = 2.0, 3.0
    dist = Kumaraswamy(a, b)
    samples = dist.sample(
        seed=jax.random.PRNGKey(1), sample_shape=(50_000,)).astype(jnp.float64)
    np.testing.assert_allclose(
        float(samples.mean()), float(dist.mean()), rtol=0.05)

  # ── jit compatibility ─────────────────────────────────────────────────────────

  def test_log_prob_jit(self):
    dist = Kumaraswamy(2.0, 3.0)
    log_prob_jit = jax.jit(dist.log_prob)
    x = jnp.array(0.5)
    expected = float(tfd.Kumaraswamy(2.0, 3.0).log_prob(x))
    np.testing.assert_allclose(float(log_prob_jit(x)), expected, rtol=1e-4)

  def test_mean_jit(self):
    dist = Kumaraswamy(2.0, 3.0)
    mean_jit = jax.jit(dist.mean)
    expected = float(tfd.Kumaraswamy(2.0, 3.0).mean())
    np.testing.assert_allclose(float(mean_jit()), expected, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
