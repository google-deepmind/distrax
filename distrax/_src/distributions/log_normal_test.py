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
"""Tests for the LogNormal distribution."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions.log_normal import LogNormal
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp_stats
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
RTOL = 1e-5


class LogNormalTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._loc   = np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32)
    self._scale = np.array([1.0, 0.5,  2.0, 0.3], dtype=np.float32)

  def test_properties(self):
    dist = LogNormal(self._loc, self._scale)
    self.assertEqual(dist.batch_shape, (4,))
    self.assertEqual(dist.event_shape, ())
    np.testing.assert_array_equal(dist.loc, self._loc)
    np.testing.assert_array_equal(dist.scale, self._scale)

  # ── log_prob ─────────────────────────────────────────────────────────────────

  @chex.all_variants()
  def test_log_prob_matches_tfp(self):
    dist = LogNormal(self._loc, self._scale)
    tfp_dist = tfd.LogNormal(self._loc, self._scale)
    xs = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    for x in xs:
      np.testing.assert_allclose(
          self.variant(dist.log_prob)(x),
          tfp_dist.log_prob(x), rtol=1e-4)

  def test_log_prob_matches_scipy(self):
    loc, scale = 0.5, 1.0
    dist = LogNormal(loc, scale)
    xs = np.array([0.1, 1.0, 2.0, 5.0], dtype=np.float64)
    np.testing.assert_allclose(
        dist.log_prob(xs.astype(np.float32)),
        sp_stats.lognorm.logpdf(xs, s=scale, scale=np.exp(loc)),
        rtol=1e-4)

  # ── mean / variance / mode / median ──────────────────────────────────────────

  def test_mean_matches_tfp(self):
    dist = LogNormal(self._loc, self._scale)
    np.testing.assert_allclose(
        dist.mean(), tfd.LogNormal(self._loc, self._scale).mean(), rtol=1e-4)

  def test_variance_matches_tfp(self):
    dist = LogNormal(self._loc, self._scale)
    np.testing.assert_allclose(
        dist.variance(), tfd.LogNormal(self._loc, self._scale).variance(),
        rtol=1e-4)

  def test_mode(self):
    """mode = exp(mu - sigma²)."""
    loc, scale = 1.0, 0.5
    dist = LogNormal(loc, scale)
    expected = float(np.exp(loc - scale ** 2))
    np.testing.assert_allclose(float(dist.mode()), expected, rtol=RTOL)

  def test_median(self):
    """median = exp(mu)."""
    loc, scale = 1.0, 0.5
    dist = LogNormal(loc, scale)
    np.testing.assert_allclose(float(dist.median()), float(np.exp(loc)),
                               rtol=RTOL)

  # ── entropy ───────────────────────────────────────────────────────────────────

  def test_entropy_matches_tfp(self):
    dist = LogNormal(self._loc, self._scale)
    np.testing.assert_allclose(
        dist.entropy(), tfd.LogNormal(self._loc, self._scale).entropy(),
        rtol=1e-4)

  # ── KL divergence ─────────────────────────────────────────────────────────────

  def test_kl_divergence_self_is_zero(self):
    dist = LogNormal(self._loc, self._scale)
    np.testing.assert_allclose(dist.kl_divergence(dist), np.zeros(4), atol=1e-6)

  def test_kl_divergence_analytic_matches_tfp(self):
    d1 = LogNormal(np.array([0.0, 1.0], dtype=np.float32),
                   np.array([1.0, 0.5], dtype=np.float32))
    d2 = LogNormal(np.array([0.5, 0.0], dtype=np.float32),
                   np.array([2.0, 1.0], dtype=np.float32))
    tfp1 = tfd.LogNormal(d1.loc, d1.scale)
    tfp2 = tfd.LogNormal(d2.loc, d2.scale)
    np.testing.assert_allclose(
        d1.kl_divergence(d2), tfd.kl_divergence(tfp1, tfp2), rtol=1e-4)

  # ── sampling ─────────────────────────────────────────────────────────────────

  def test_sample_shape(self):
    dist = LogNormal(self._loc, self._scale)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(10,))
    self.assertEqual(samples.shape, (10, 4))

  def test_sample_positive(self):
    dist = LogNormal(0.0, 1.0)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(1000,))
    self.assertTrue(jnp.all(samples > 0))

  def test_sample_moments(self):
    """Empirical mean should be close to analytical mean."""
    loc, scale = 0.5, 0.5
    dist = LogNormal(loc, scale)
    samples = dist.sample(
        seed=jax.random.PRNGKey(1), sample_shape=(50_000,)).astype(jnp.float64)
    np.testing.assert_allclose(
        float(samples.mean()), float(dist.mean()), rtol=0.05)

  # ── jit compatibility ─────────────────────────────────────────────────────────

  def test_log_prob_jit(self):
    dist = LogNormal(0.5, 1.0)
    log_prob_jit = jax.jit(dist.log_prob)
    x = jnp.array(2.0)
    np.testing.assert_allclose(
        float(log_prob_jit(x)),
        float(tfd.LogNormal(0.5, 1.0).log_prob(x)), rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
