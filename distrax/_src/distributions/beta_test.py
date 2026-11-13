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
"""Tests for `beta.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import beta as base_beta
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np


class BetaTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(base_beta.Beta)

  @parameterized.named_parameters(
      ('0d params', (), (), ()),
      ('1d params', (2,), (2,), (2,)),
      ('2d params, no broadcast', (3, 2), (3, 2), (3, 2)),
      ('2d params, broadcasted alpha', (2,), (3, 2), (3, 2)),
      ('2d params, broadcasted beta', (3, 2), (2,), (3, 2)),
  )
  def test_properties(self, shape_of_alpha, shape_of_beta, batch_shape):
    rng = np.random.default_rng(42)
    alpha = 0.1 + rng.uniform(size=shape_of_alpha)
    beta = 0.1 + rng.uniform(size=shape_of_beta)
    dist = base_beta.Beta(alpha, beta)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=1e-3)(
        dist.alpha, np.broadcast_to(alpha, batch_shape))
    self.assertion_fn(rtol=1e-3)(dist.beta, np.broadcast_to(beta, batch_shape))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std beta, no shape', (1, 1), ()),
      ('1d std beta, int shape', (1, 1), 1),
      ('1d std beta, 1-tuple shape', (1, 1), (1,)),
      ('1d std beta, 2-tuple shape', (1, 1), (2, 2)),
      ('2d std beta, no shape', (np.ones(2), np.ones(2)), ()),
      ('2d std beta, int shape', ([1, 1], [1, 1]), 1),
      ('2d std beta, 1-tuple shape', (np.ones(2), np.ones(2)), (1,)),
      ('2d std beta, 2-tuple shape', ([1, 1], [1, 1]), (2, 2)),
      ('rank 2 std beta, 2-tuple shape', (np.ones((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted alpha', (1, np.ones(3)), (2, 2)),
      ('broadcasted beta', (np.ones(3), 1), ()),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, float16', 'sample', jnp.float16),
      ('sample, float32', 'sample', jnp.float32),
      ('sample_and_log_prob, float16', 'sample_and_log_prob', jnp.float16),
      ('sample_and_log_prob, float32', 'sample_and_log_prob', jnp.float32),
  )
  def test_sample_dtype(self, method, dtype):
    dist = self.distrax_cls(alpha=jnp.ones((), dtype), beta=jnp.ones((), dtype))
    # pyrefly: ignore[missing-attribute]
    samples = self.variant(getattr(dist, method))(seed=self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.dtype, dist.dtype)
    self.assertEqual(samples.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample', 'sample'),
      ('sample_and_log_prob', 'sample_and_log_prob'),
  )
  def test_sample_values(self, method):
    rng = np.random.default_rng(42)
    alpha = jnp.array(np.abs(rng.normal(size=(4, 3, 2))))
    beta = jnp.array(np.abs(rng.normal(size=(4, 3, 2))))
    n_samples = 100000
    dist = self.distrax_cls(alpha, beta)
    sample_fn = self.variant(  # pyrefly: ignore[missing-attribute]
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples))
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.shape, (n_samples,) + (4, 3, 2))
    self.assertTrue(np.all(np.logical_and(samples >= 0., samples <= 1.)))
    self.assertion_fn(rtol=0.1)(np.mean(samples, axis=0), dist.mean())
    self.assertion_fn(rtol=0.1)(np.std(samples, axis=0), dist.stddev())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std beta, no shape', (11, 3), ()),
      ('1d std beta, int shape', (11, 3), 1),
      ('1d std beta, 1-tuple shape', (11, 3), (1,)),
      ('1d std beta, 2-tuple shape', (1, 1), (2, 2)),
      ('2d std beta, no shape', (np.ones(2), np.ones(2)), ()),
      ('2d std beta, int shape', ([1, 1], [1, 1]), 1),
      ('2d std beta, 1-tuple shape', (np.ones(2), np.ones(2)), (1,)),
      ('2d std beta, 2-tuple shape', ([1, 1], [1, 1]), (2, 2)),
      ('rank 2 std beta, 2-tuple shape', (np.ones((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted alpha', (1, np.ones(3)), (2, 2)),
      ('broadcasted beta', (np.ones(3), 1), ()),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_and_log_prob(
        dist_args=distr_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (3.1, 1), 0.6),
      ('1d dist, 2d value', (0.5, 0.1), np.array([0.3, 0.8])),
      ('2d dist, 1d value', (0.5 + np.zeros(2), 0.3 * np.ones(2)), 0.7),
      ('2d broadcasted dist, 1d value', (0.4 + np.zeros(2), 0.8), 0.7),
      ('2d dist, 2d value',
       ([0.1, 0.5], 0.9 * np.ones(2)), np.array([0.2, 0.7])),
      ('edge cases alpha=1', (1., np.array([0.5, 2.])), np.array([0., 1.])),
      ('edge cases beta=1', (np.array([0.5, 2.]), 1.), np.array([0., 1.])),
  )
  def test_methods_with_value(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    for method in ['prob', 'log_prob', 'cdf', 'log_cdf', 'survival_function',
                   'log_survival_function']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @parameterized.parameters((jnp.float16, 1000.), (jnp.float32, 1e7))
  def test_log_prob_unbalanced_parameters(self, dtype, large):
    beta = jnp.array([large, large / 2.], dtype=dtype)
    value = jnp.array([[1. / large], [2. / large]], dtype=dtype)
    # Evaluate the exact Beta(1, b) identity at the represented inputs.
    b64, x64 = np.asarray(beta, np.float64), np.asarray(value, np.float64)
    expected = np.log(b64) + (b64 - 1.) * np.log1p(-x64)
    log_prob = lambda b, x: base_beta.Beta(jnp.array(1., dtype), b).log_prob(x)
    for fn in [log_prob, jax.jit(log_prob)]:
      actual = fn(beta, value)
      self.assertEqual(actual.dtype, dtype)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-5)

  def test_log_prob_support_boundaries(self):
    alpha = jnp.array([1., 1., 2., 0.5, 2., 1., 1.])
    beta = jnp.array([1., 2., 1., 2., 0.5, 1., 1.])
    value = jnp.array([0., 0., 0., 0., 1., 1., -0.1])
    expected = np.array([0., np.log(2.), -np.inf, np.inf, np.inf, 0., np.nan])
    log_prob = lambda a, b, x: base_beta.Beta(a, b).log_prob(x)
    for fn in [log_prob, jax.jit(log_prob)]:
      np.testing.assert_allclose(fn(alpha, beta, value), expected, atol=2e-6)

  @parameterized.parameters(jnp.uint8, jnp.uint32, jnp.bool_)
  def test_log_prob_nonfloating_values(self, dtype):
    value = jnp.array([0, 1], dtype=dtype)
    log_prob = lambda x: base_beta.Beta(2., 2.).log_prob(x)
    for fn in [log_prob, jax.jit(log_prob)]:
      np.testing.assert_array_equal(fn(value), [-np.inf, -np.inf])

  def test_log_prob_interior_gradients(self):
    alpha, beta, value = 2., 3., 0.25
    log_prob = lambda a, b, x: base_beta.Beta(a, b).log_prob(x)
    # Integer-shape digamma differences give exact harmonic sums.
    expected = (np.log(value) + 1. / 2. + 1. / 3. + 1. / 4.,
                np.log1p(-value) + 1. / 3. + 1. / 4.,
                (alpha - 1.) / value - (beta - 1.) / (1. - value))
    grad_fn = jax.grad(log_prob, argnums=(0, 1, 2))
    for fn in [grad_fn, jax.jit(grad_fn)]:
      np.testing.assert_allclose(fn(alpha, beta, value), expected, atol=2e-6)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('no broadcast', ([0.1, 1.3, 0.5], [0.5, 1.3, 1.5])),
      ('broadcasted alpha', (0.5, [0.5, 1.3, 1.5])),
      ('broadcasted beta', ([0.1, 1.3, 0.5], 0.8)),
  )
  def test_method(self, distr_params):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    for method in ['entropy', 'mean', 'variance', 'stddev']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=distr_params,
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('alpha>1, beta>1, no broadcast', 1.5, 2.5, 0.25),
      ('alpha>1, beta>1, broadcasted alpha', 1.5, [2.5, 5.5], [0.25, 0.1]),
      ('alpha>1, beta>1, broadcasted beta', [1.5, 4.5], 2.5, [0.25, 0.7]),
      ('alpha<1, beta<1', 0.5, 0.1, np.nan),
      ('alpha=1, beta=1', 1., 1., np.nan),
      ('alpha=1, beta>1', 1., 1.5, 0.),
      ('alpha<1, beta>1', 0.5, 1.5, 0.),
      ('alpha>1, beta=1', 1.5, 1., 1.),
      ('alpha>1, beta<1', 1.5, 0.5, 1.),
  )
  def test_mode(self, alpha, beta, expected_result):
    dist = self.distrax_cls(alpha, beta)
    result = self.variant(dist.mode)()  # pyrefly: ignore[missing-attribute]
    if np.any(np.isnan(expected_result)):
      self.assertTrue(jnp.isnan(result))
    else:
      self.assertion_fn(rtol=1e-3)(result, expected_result)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('distrax_to_distrax', 'distrax_to_distrax'),
      ('distrax_to_tfp', 'distrax_to_tfp'),
      ('tfp_to_distrax', 'tfp_to_distrax'),
  )
  def test_with_two_distributions(self, mode_string):
    rng = np.random.default_rng(42)
    alpha1 = np.abs(rng.normal(size=(4, 1, 2))).astype(np.float32)
    beta1 = np.abs(rng.normal(size=(4, 3, 2))).astype(np.float32)
    alpha2 = np.abs(rng.normal(size=(3, 2))).astype(np.float32)
    beta2 = np.abs(rng.normal(size=(3, 2))).astype(np.float32)
    for method in ['kl_divergence', 'cross_entropy']:
      with self.subTest(method=method):
        super()._test_with_two_distributions(
            attribute_string=method,
            mode_string=mode_string,
            dist1_kwargs={'alpha': alpha1, 'beta': beta1},
            dist2_kwargs={'alpha': alpha2, 'beta': beta2},
            tfp_dist1_kwargs={
                'concentration1': alpha1, 'concentration0': beta1},
            tfp_dist2_kwargs={
                'concentration1': alpha2, 'concentration0': beta2},
            assertion_fn=self.assertion_fn(rtol=3e-2))

  def test_jitable(self):
    super()._test_jittable(
        (0.1, 1.5), assertion_fn=self.assertion_fn(rtol=1e-3))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(42)
    alpha = jnp.array(np.abs(rng.normal(size=(4, 3, 2))))
    beta = jnp.array(np.abs(rng.normal(size=(4, 3, 2))))
    dist = self.distrax_cls(alpha, beta)
    self.assertion_fn(rtol=1e-3)(dist[slice_].alpha, alpha[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].beta, beta[slice_])

  def test_slice_different_parameterization(self):
    rng = np.random.default_rng(42)
    alpha = np.abs(rng.normal(size=(4, 3, 2)))
    beta = np.abs(rng.normal(size=(3, 2)))
    dist = self.distrax_cls(alpha, beta)
    self.assertion_fn(rtol=1e-3)(dist[0].alpha, alpha[0])
    self.assertion_fn(rtol=1e-3)(dist[0].beta, beta)  # Not slicing beta.


if __name__ == '__main__':
  absltest.main()
