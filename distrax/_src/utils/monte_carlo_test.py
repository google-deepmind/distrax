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
"""Tests for `monte_carlo.py`."""

from absl.testing import absltest

import chex
from distrax._src.distributions import mvn_diag
from distrax._src.distributions import normal
from distrax._src.utils import monte_carlo
import haiku as hk
import jax
import numpy as np
from tensorflow_probability.substrates import jax as tfp


class McTest(absltest.TestCase):

  def test_estimate_kl_with_dice(self):
    batch_size = 5
    num_actions = 11
    num_samples = 1024
    rng_seq = hk.PRNGSequence(0)

    distribution_a = tfp.distributions.Categorical(
        logits=jax.random.normal(next(rng_seq), [batch_size, num_actions]))
    distribution_b = tfp.distributions.Categorical(
        logits=jax.random.normal(next(rng_seq), [batch_size, num_actions]))

    kl_estim_exact = monte_carlo.estimate_kl_best_effort(
        distribution_a, distribution_b, next(rng_seq), num_samples=num_samples)
    kl_estim_mc = monte_carlo.mc_estimate_kl(
        distribution_a, distribution_b, next(rng_seq), num_samples=num_samples)
    kl = distribution_a.kl_divergence(distribution_b)
    np.testing.assert_allclose(kl, kl_estim_exact, rtol=1e-5)
    np.testing.assert_allclose(kl, kl_estim_mc, rtol=2e-1)

  def test_estimate_continuous_kl_with_dice(self):
    _check_kl_estimator(monte_carlo.mc_estimate_kl, tfp.distributions.Normal)
    _check_kl_estimator(monte_carlo.mc_estimate_kl, normal.Normal)

  def test_estimate_continuous_kl_with_reparameterized(self):
    _check_kl_estimator(monte_carlo.mc_estimate_kl_with_reparameterized,
                        tfp.distributions.Normal)
    _check_kl_estimator(monte_carlo.mc_estimate_kl_with_reparameterized,
                        normal.Normal)

  def test_estimate_mode(self):
    with self.subTest('ScalarEventShape'):
      distribution = normal.Normal(
          loc=np.zeros((4, 5, 100)),
          scale=np.ones((4, 5, 100)))
      mode_estimate = monte_carlo.mc_estimate_mode(
          distribution, rng_key=42, num_samples=100)
      mean_mode_estimate = np.abs(np.mean(mode_estimate))
      self.assertLess(mean_mode_estimate, 1e-3)
    with self.subTest('NonScalarEventShape'):
      distribution = mvn_diag.MultivariateNormalDiag(
          loc=np.zeros((4, 5, 100)),
          scale_diag=np.ones((4, 5, 100)))
      mv_mode_estimate = monte_carlo.mc_estimate_mode(
          distribution, rng_key=42, num_samples=100)
      mean_mv_mode_estimate = np.abs(np.mean(mv_mode_estimate))
      self.assertLess(mean_mv_mode_estimate, 1e-1)
      # The mean of the mode-estimate of the Normal should be a lot closer
      # to 0 compared to the MultivariateNormal, because the 100 less samples
      # are taken and most of the mass in a high-dimensional gaussian is NOT
      # at 0!
      self.assertLess(10 * mean_mode_estimate, mean_mv_mode_estimate)


def _check_kl_estimator(estimator_fn, distribution_fn, num_samples=10000,
                        rtol=1e-1, atol=1e-3, grad_rtol=2e-1, grad_atol=1e-1):
  """Compares the estimator_fn output and gradient to exact KL."""
  rng_key = jax.random.PRNGKey(0)

  def expected_kl(params):
    distribution_a = distribution_fn(**params[0])
    distribution_b = distribution_fn(**params[1])
    return distribution_a.kl_divergence(distribution_b)

  def estimate_kl(params):
    distribution_a = distribution_fn(**params[0])
    distribution_b = distribution_fn(**params[1])
    return estimator_fn(distribution_a, distribution_b, rng_key=rng_key,
                        num_samples=num_samples)

  params = (
      dict(loc=0.0, scale=1.0),
      dict(loc=0.1, scale=1.0),
  )
  expected_value, expected_grad = jax.value_and_grad(expected_kl)(params)
  value, grad = jax.value_and_grad(estimate_kl)(params)

  np.testing.assert_allclose(expected_value, value, rtol=rtol, atol=atol)
  chex.assert_tree_all_close(expected_grad, grad, rtol=grad_rtol,
                             atol=grad_atol)


if __name__ == '__main__':
  absltest.main()
