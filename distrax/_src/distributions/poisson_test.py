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
"""Tests for the Poisson distribution."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions.poisson import Poisson
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp_stats


RTOL = 1e-5


class PoissonTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rate = np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float32)

  # ── construction ─────────────────────────────────────────────────────────────

  def test_properties(self):
    dist = Poisson(rate=self._rate)
    self.assertEqual(dist.batch_shape, (4,))
    self.assertEqual(dist.event_shape, ())
    np.testing.assert_array_equal(dist.rate, self._rate)

  # ── log_prob ─────────────────────────────────────────────────────────────────

  @chex.all_variants()
  @parameterized.named_parameters(
      ('k0', 0), ('k1', 1), ('k3', 3), ('k10', 10))
  def test_log_prob(self, k):
    dist = Poisson(rate=self._rate)
    k_arr = jnp.full(self._rate.shape, k)
    log_prob = self.variant(dist.log_prob)(k_arr)
    expected = sp_stats.poisson.logpmf(k, mu=self._rate)
    np.testing.assert_allclose(log_prob, expected, rtol=RTOL)

  def test_log_prob_batched_k(self):
    dist = Poisson(rate=2.0)
    ks = jnp.arange(10, dtype=jnp.float32)
    log_probs = dist.log_prob(ks)
    expected = sp_stats.poisson.logpmf(np.arange(10), mu=2.0)
    np.testing.assert_allclose(log_probs, expected, rtol=RTOL)

  # ── mean / variance / mode ───────────────────────────────────────────────────

  def test_mean(self):
    dist = Poisson(rate=self._rate)
    np.testing.assert_allclose(dist.mean(), self._rate, rtol=RTOL)

  def test_variance(self):
    dist = Poisson(rate=self._rate)
    np.testing.assert_allclose(dist.variance(), self._rate, rtol=RTOL)

  def test_mode(self):
    dist = Poisson(rate=np.array([0.5, 1.0, 1.9, 2.0, 3.7], dtype=np.float32))
    # mode = floor(λ) for non-integers, λ-1 for integers (both floor(λ) and
    # λ-1 are valid for integer λ; floor(λ) is our convention).
    expected = np.array([0, 1, 1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(dist.mode(), expected)

  # ── entropy ──────────────────────────────────────────────────────────────────

  def test_entropy(self):
    dist = Poisson(rate=self._rate)
    expected = sp_stats.poisson.entropy(mu=self._rate)
    np.testing.assert_allclose(dist.entropy(), expected, rtol=1e-3)

  # ── KL divergence ────────────────────────────────────────────────────────────

  def test_kl_divergence_against_scipy(self):
    r1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    r2 = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    dist1 = Poisson(rate=r1)
    dist2 = Poisson(rate=r2)
    kl = dist1.kl_divergence(dist2)
    # scipy does not have Poisson KL; compare against analytic formula:
    # KL = r1*log(r1/r2) + r2 - r1
    expected = r1 * np.log(r1 / r2) + r2 - r1
    np.testing.assert_allclose(kl, expected, rtol=RTOL)

  def test_kl_divergence_self_is_zero(self):
    dist = Poisson(rate=np.array([0.5, 2.0, 5.0], dtype=np.float32))
    kl = dist.kl_divergence(dist)
    np.testing.assert_allclose(kl, np.zeros(3), atol=1e-6)

  # ── sampling ─────────────────────────────────────────────────────────────────

  def test_sample_shape(self):
    dist = Poisson(rate=self._rate)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(100,))
    self.assertEqual(samples.shape, (100, 4))

  def test_sample_nonnegative(self):
    dist = Poisson(rate=2.0)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(1000,))
    self.assertTrue(jnp.all(samples >= 0))

  def test_sample_moments(self):
    """Empirical mean and variance should be close to λ (large-sample check)."""
    lam = 3.0
    dist = Poisson(rate=lam)
    samples = dist.sample(
        seed=jax.random.PRNGKey(1), sample_shape=(50_000,)).astype(jnp.float32)
    np.testing.assert_allclose(float(samples.mean()), lam, rtol=0.05)
    np.testing.assert_allclose(float(samples.var()), lam, rtol=0.05)

  # ── dtype ─────────────────────────────────────────────────────────────────────

  def test_integer_dtype_samples(self):
    dist = Poisson(rate=2.0, dtype=jnp.int32)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(5,))
    self.assertEqual(samples.dtype, jnp.int32)

  def test_float_dtype_samples(self):
    dist = Poisson(rate=2.0, dtype=jnp.float32)
    samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(5,))
    self.assertEqual(samples.dtype, jnp.float32)

  # ── jit compatibility ─────────────────────────────────────────────────────────

  def test_log_prob_jit(self):
    dist = Poisson(rate=2.0)
    log_prob_jit = jax.jit(dist.log_prob)
    k = jnp.array(3.0)
    np.testing.assert_allclose(
        float(log_prob_jit(k)),
        float(sp_stats.poisson.logpmf(3, mu=2.0)),
        rtol=RTOL)

  def test_kl_divergence_jit(self):
    d1 = Poisson(rate=1.0)
    d2 = Poisson(rate=2.0)
    kl = jax.jit(d1.kl_divergence)(d2)
    expected = 1.0 * np.log(1.0 / 2.0) + 2.0 - 1.0
    np.testing.assert_allclose(float(kl), expected, rtol=RTOL)


if __name__ == '__main__':
  absltest.main()
