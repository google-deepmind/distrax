# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `von_mises.py`."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import von_mises
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp_stats
import scipy.special
from tensorflow_probability.substrates import jax as tfp


class VonMisesTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(von_mises.VonMises)
    self.loc = np.reshape(np.linspace(-5., 5., 7), [-1, 1])
    self.concentration = np.reshape(np.logspace(-3., 3., 7), [1, -1])
    self.rng = np.random.default_rng(317070)

  @parameterized.named_parameters(
      ('1d std vonmises', (0, 1)),
      ('2d std vonmises', (np.zeros(2), np.ones(2))),
      ('rank 2 std vonmises', (np.zeros((3, 2)), np.ones((3, 2)))),
      ('broadcasted loc', (0, np.ones(3))),
      ('broadcasted concentration', (np.ones(3), 1)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @parameterized.named_parameters(
      ('0d concentration', ()),
      ('1d concentration', (4,)),
      ('2d concentration', (3, 4)),
  )
  def test_loc_shape_properties(self, shape):
    loc = self.rng.uniform()
    concentration = self.rng.uniform(size=shape)
    dist = self.distrax_cls(loc=loc, concentration=concentration)
    self.assertion_fn(rtol=1e-3)(dist.loc, loc)
    self.assertion_fn(rtol=1e-3)(dist.concentration, concentration)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, shape)

  @parameterized.named_parameters(
      ('0d loc', ()),
      ('1d loc', (4,)),
      ('2d loc', (3, 4)),
  )
  def test_concentration_shape_properties(self, shape):
    loc = self.rng.uniform(size=shape)
    concentration = self.rng.uniform()
    dist = self.distrax_cls(loc=loc, concentration=concentration)
    self.assertion_fn(rtol=1e-3)(dist.loc, loc)
    self.assertion_fn(rtol=1e-3)(dist.concentration, concentration)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std vonmises, no shape', (0, 1), ()),
      ('1d std vonmises, int shape', (0, 1), 1),
      ('1d std vonmises, 1-tuple shape', (0, 1), (1,)),
      ('1d std vonmises, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std vonmises, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std vonmises, int shape', ([0, 0], [1, 1]), 1),
      ('2d std vonmises, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std vonmises, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
      ('rank 2 std vonmises, 2-tuple shape', (np.zeros((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted loc', (0, np.ones(3)), (2, 2)),
      ('broadcasted scale', (np.ones(3), 1), ()),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std vonmises, no shape', (0, 1), ()),
      ('1d std vonmises, int shape', (0, 1), 1),
      ('1d std vonmises, 1-tuple shape', (0, 1), (1,)),
      ('1d std vonmises, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std vonmises, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std vonmises, int shape', ([0, 0], [1, 1]), 1),
      ('2d std vonmises, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std vonmises, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
      ('rank 2 std vonmises, 2-tuple shape', (np.zeros((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted loc', (0, np.ones(3)), (2, 2)),
      ('broadcasted scale', (np.ones(3), 1), ()),
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
      ('sample, float16', 'sample', jnp.float16),
      ('sample, float32', 'sample', jnp.float32),
      ('sample_and_log_prob, float16', 'sample_and_log_prob', jnp.float16),
      ('sample_and_log_prob, float32', 'sample_and_log_prob', jnp.float32),
  )
  def test_sample_dtype(self, method, dtype):
    dist = self.distrax_cls(
        loc=self.loc.astype(dtype),
        concentration=self.concentration.astype(dtype),
    )
    samples = self.variant(getattr(dist, method))(seed=self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.dtype, dist.dtype)
    self.assertEqual(samples.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0., 1.), np.array([1, 2])),
      ('2d dist, 1d value', (np.zeros(2), np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 1), 1),
      ('2d dist, 2d value', (np.zeros(2), np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), np.pi),
  )
  def test_method_with_input(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    for method in [
        'log_prob', 'prob', 'cdf', 'log_cdf', 'survival_function',
        'log_survival_function'
    ]:
      with self.subTest(method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=distr_params,
            dist_kwargs={},
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', (0., 1.), 'entropy'),
      ('mean', (0, 1), 'mean'),
      ('mean from 1d params', ([-1, 1], [1, 2]), 'mean'),
      ('variance', (0, 1), 'variance'),
      ('variance from np params', (np.ones(2), np.ones(2)), 'variance'),
      ('stddev', (0, 1), 'stddev'),
      ('stddev from rank 2 params', (np.ones((2, 3)), np.ones(
          (2, 3))), 'stddev'),
      ('mode', (0, 1), 'mode'),
  )
  def test_method(self, distr_params, function_string):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_attribute(
        function_string,
        distr_params,
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_two_distributions(self, function_string, mode_string):
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'loc': self.rng.standard_normal((4, 1, 2)),
            'concentration': np.asarray([[0.8, 0.2], [0.1, 1.2], [1.4, 3.1]]),
        },
        dist2_kwargs={
            'loc': self.rng.standard_normal((3, 2)),
            'concentration': 0.1 + self.rng.standard_normal((4, 1, 2)),
        },
        assertion_fn=self.assertion_fn(rtol=1e-2))

  def test_jittable(self):
    super()._test_jittable(
        (np.zeros((3,)), np.ones((3,))),
        assertion_fn=functools.partial(
            np.testing.assert_allclose, rtol=1e-04, atol=1e-04
        )
    )

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    loc = jnp.array(self.rng.standard_normal((3, 4, 5)))
    concentration = jnp.array(self.rng.standard_normal((3, 4, 5)))
    dist = self.distrax_cls(loc=loc, concentration=concentration)
    self.assertion_fn(rtol=1e-2)(dist[slice_].mean(), loc[slice_])

  def test_slice_different_parameterization(self):
    loc = jnp.array(self.rng.standard_normal((4)))
    concentration = jnp.array(self.rng.standard_normal((3, 4)))
    dist = self.distrax_cls(loc=loc, concentration=concentration)
    self.assertion_fn(rtol=1e-2)(dist[0].mean(), loc)  # Not slicing loc.
    self.assertion_fn(rtol=1e-2)(dist[0].concentration, concentration[0])

  def test_von_mises_log_pdf(self):
    locs_v = .1
    concentrations_v = .2
    x = np.array([2., 3., 4., 5., 6., 7.])
    vm = self.distrax_cls(locs_v, concentrations_v)
    expected_log_prob = sp_stats.vonmises.logpdf(  # pytype: disable=module-attr
        x,
        concentrations_v,
        loc=locs_v
    )
    log_prob = vm.log_prob(x)
    np.testing.assert_allclose(
        expected_log_prob, log_prob, rtol=1e-04, atol=1e-04
    )

  def test_von_mises_log_pdf_uniform(self):
    x = np.array([2., 3., 4., 5., 6., 7.])
    vm = self.distrax_cls(.1, 0.)
    log_prob = vm.log_prob(x)
    expected_log_prob = np.array([-np.log(2. * np.pi)] * 6)
    np.testing.assert_allclose(
        expected_log_prob, log_prob, rtol=1e-04, atol=1e-04
    )

  def test_von_mises_pdf(self):
    locs_v = .1
    concentrations_v = .2
    x = np.array([2., 3., 4., 5., 6., 7.])
    vm = self.distrax_cls(locs_v, concentrations_v)
    prob = vm.prob(x)
    expected_prob = sp_stats.vonmises.pdf(  # pytype: disable=module-attr
        x, concentrations_v, loc=locs_v
    )
    np.testing.assert_allclose(expected_prob, prob, rtol=1e-04, atol=1e-04)

  def test_von_mises_pdf_uniform(self):
    x = np.array([2., 3., 4., 5., 6., 7.])
    vm = self.distrax_cls(1., 0.)
    prob = vm.prob(x)
    expected_prob = np.array([1. / (2. * np.pi)] * 6)
    np.testing.assert_allclose(expected_prob, prob, rtol=1.5e-7, atol=1e-7)

  def test_von_mises_cdf(self):
    # We follow the scipy definition for cdf when loc=0 and x is in [-pi, pi].
    locs_v = np.zeros(shape=(7, 1, 1))
    concentrations_v = np.reshape(np.logspace(-3., 3., 7), [1, -1, 1])
    x = np.reshape(np.linspace(-np.pi, np.pi, 7), [1, 1, -1])
    vm = self.distrax_cls(locs_v, concentrations_v)
    cdf = vm.cdf(x)
    expected_cdf = sp_stats.vonmises.cdf(  # pytype: disable=module-attr
        x, concentrations_v, loc=locs_v
    )
    np.testing.assert_allclose(expected_cdf, cdf, atol=1e-4, rtol=1e-4)

  def test_von_mises_survival_function(self):
    locs_v = np.reshape(np.linspace(-5, 5, 7), [-1, 1, 1])
    concentrations_v = np.reshape(np.logspace(-3., 3., 7), [1, -1, 1])
    x = np.reshape(np.linspace(-5, 5, 7), [1, 1, -1])
    vm = self.distrax_cls(locs_v, concentrations_v)
    cdf = vm.cdf(x)
    surv = vm.survival_function(x)
    np.testing.assert_allclose(surv, 1 - cdf, atol=1e-4, rtol=1e-4)

  def test_von_mises_cdf_out_of_bounds(self):
    locs_v = np.reshape(np.linspace(-np.pi, np.pi, 7), [-1, 1, 1])
    concentrations_v = np.reshape(np.logspace(-3., 3., 7), [1, -1, 1])
    vm = self.distrax_cls(locs_v, concentrations_v)
    x = np.linspace(-5 * np.pi, -np.pi, 7)
    cdf = vm.cdf(x)
    expected_cdf = 0.
    np.testing.assert_allclose(expected_cdf, cdf, rtol=1.5e-7, atol=1e-7)

    x = np.linspace(np.pi, 5 * np.pi, 7)
    cdf = vm.cdf(x)
    expected_cdf = 1.
    np.testing.assert_allclose(expected_cdf, cdf, rtol=1.5e-7, atol=1e-7)

  def test_von_mises_log_cdf(self):
    locs_v = np.reshape(np.linspace(-5, 5, 7), [-1, 1, 1])
    concentrations_v = np.reshape(np.logspace(-3., 3., 7), [1, -1, 1])
    x = np.reshape(np.linspace(-5, 5, 7), [1, 1, -1])
    vm = self.distrax_cls(locs_v, concentrations_v)
    cdf = vm.cdf(x)
    logcdf = vm.log_cdf(x)
    np.testing.assert_allclose(np.log(cdf), logcdf, atol=1e-4, rtol=1e-4)

  def test_von_mises_log_survival(self):
    locs_v = np.reshape(np.linspace(-5, 5, 7), [-1, 1, 1])
    concentrations_v = np.reshape(np.logspace(-3., 3., 7), [1, -1, 1])
    x = np.reshape(np.linspace(-5, 5, 7), [1, 1, -1])
    vm = self.distrax_cls(locs_v, concentrations_v)
    surv = vm.survival_function(x)
    logsurv = vm.log_survival_function(x)
    np.testing.assert_allclose(np.log(surv), logsurv, atol=1e-4, rtol=1e-4)

  def test_von_mises_cdf_uniform(self):
    x = np.linspace(-np.pi, np.pi, 7)
    vm = self.distrax_cls(0., 0.)
    cdf = vm.cdf(x)
    expected_cdf = (x + np.pi) / (2. * np.pi)
    np.testing.assert_allclose(expected_cdf, cdf, rtol=1.5e-7, atol=1e-7)

  def test_von_mises_cdf_gradient_simple(self):
    n = 10
    locs = jnp.array([1.] * n)
    concentrations = np.logspace(-3, 3, n)
    x = np.linspace(-5, 5, n)
    def f(x, l, c):
      vm = self.distrax_cls(l, c)
      cdf = vm.cdf(x)
      return cdf
    jax.test_util.check_grads(f, (x, locs, concentrations), order=1)

  def test_von_mises_sample_gradient(self):
    n = 10
    locs = jnp.array([1.] * n)
    concentrations = np.logspace(-3, 3, n)
    def f(l, c):
      vm = self.distrax_cls(l, c)
      x = vm.sample(seed=1)
      return x
    jax.test_util.check_grads(
        f,
        (locs, concentrations),
        order=1,
        rtol=0.1
    )

  def test_von_mises_uniform_sample_gradient(self):
    def f(c):
      vm = self.distrax_cls(0., c)
      x = vm.sample(seed=1)
      return x
    # The finite difference is not very accurate, but the analytic gradient
    # should not be zero.
    self.assertNotEqual(jax.grad(f)(0.), 0)

  @parameterized.named_parameters(
      ('small concentration', 1.),
      ('medium concentration', 10.),
      ('large concentration', 1000.),
      ('uniform', 1e-6),
  )
  def test_von_mises_sample_gradient_comparison(self, concentration):
    # Compares the von Mises sampling gradient against the reference
    # implementation from tensorflow_probability.
    locs = 0.
    def f(seed, l, c):
      vm = self.distrax_cls(l, c)
      x = vm.sample(seed=seed)  # pylint: disable=cell-var-from-loop
      return x
    jax_sample_and_grad = jax.value_and_grad(f, argnums=2)

    def samples_grad(s, concentration):
      broadcast_concentration = concentration
      _, dcdf_dconcentration = tfp.math.value_and_gradient(
          lambda conc: tfp.distributions.von_mises.von_mises_cdf(s, conc),
          broadcast_concentration)
      inv_prob = np.exp(-concentration * (np.cos(s) - 1.)) * (
          (2. * np.pi) * scipy.special.i0e(concentration)
      )
      # Computes the implicit derivative,
      # dz = dconc * -(dF(z; conc) / dconc) / p(z; conc)
      dsamples = -dcdf_dconcentration * inv_prob
      return dsamples

    for seed in range(10):
      sample, sample_grad = jax_sample_and_grad(
          seed, jnp.array(locs), jnp.array(concentration)
      )
      comparison = samples_grad(sample, concentration)
      np.testing.assert_allclose(
          comparison, sample_grad, rtol=1e-06, atol=1e-06
      )

  def test_von_mises_sample_moments(self):
    locs_v = np.array([-1., 0.3, 2.3])
    concentrations_v = np.array([1., 2., 10.])
    vm = self.distrax_cls(locs_v, concentrations_v)

    n = 1000
    samples = vm.sample(sample_shape=(n,), seed=1)

    expected_mean = vm.mean()
    actual_mean = jnp.arctan2(
        jnp.mean(jnp.sin(samples), axis=0),
        jnp.mean(jnp.cos(samples), axis=0),
    )

    expected_variance = vm.variance()
    standardized_samples = samples - vm.mean()
    variance_samples = jnp.mean(1. - jnp.cos(standardized_samples), axis=0)

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.1)
    np.testing.assert_allclose(
        variance_samples, expected_variance, rtol=0.1
    )

  def test_von_mises_sample_variance_uniform(self):
    vm = self.distrax_cls(1., 0.)

    n = 1000
    samples = vm.sample(sample_shape=(n,), seed=1)

    # For circular uniform distribution, the mean is not well-defined,
    # so only checking the variance.
    expected_variance = 1.
    standardized_samples = samples - vm.mean()
    variance_samples = jnp.mean(1. - jnp.cos(standardized_samples), axis=0)

    np.testing.assert_allclose(
        variance_samples, expected_variance, rtol=0.1
    )

  def test_von_mises_sample_extreme_concentration(self):
    loc = jnp.array([1., np.nan, 1., 1., np.nan])
    min_value = np.finfo(np.float32).min
    max_value = np.finfo(np.float32).max
    concentration = jnp.array([min_value, 1., max_value, np.nan, np.nan])
    vm = self.distrax_cls(loc, concentration)

    samples = vm.sample(seed=1)
    # Check that it does not end up in an infinite loop.
    self.assertEqual(samples.shape, (5,))

  def test_von_mises_sample_ks_test(self):
    concentrations_v = np.logspace(-3, 3, 7)
    # We are fixing the location to zero. The reason is that for loc != 0,
    # scipy's von Mises distribution CDF becomes shifted, so it's no longer
    # in [0, 1], but is in something like [-0.3, 0.7]. This breaks kstest.
    vm = self.distrax_cls(0., concentrations_v)
    n = 1000
    sample_values = vm.sample(sample_shape=(n,), seed=1)
    self.assertEqual(sample_values.shape, (n, 7))

    fails = 0
    trials = 0
    for concentrationi, concentration in enumerate(concentrations_v):
      s = sample_values[:, concentrationi]
      trials += 1
      p = sp_stats.kstest(
          s,
          sp_stats.vonmises(concentration).cdf  # pytype: disable=not-callable
      )[1]
      if p <= 0.05:
        fails += 1
    self.assertLess(fails, trials * 0.1)

  def test_von_mises_sample_uniform_ks_test(self):
    locs_v = np.linspace(-10., 10., 7)
    vm = self.distrax_cls(locs_v, 0.)

    n = 1000
    sample_values = vm.sample(sample_shape=(n,), seed=1)
    self.assertEqual(sample_values.shape, (n, 7))

    fails = 0
    trials = 0
    for loci, _ in enumerate(locs_v):
      s = sample_values[:, loci]
      # [-pi, pi] -> [0, 1]
      s = (s + np.pi) / (2. * np.pi)
      trials += 1
      # Compare to the CDF of Uniform(0, 1) random variable.
      p = sp_stats.kstest(s, sp_stats.uniform.cdf)[1]
      if p <= 0.05:
        fails += 1
    self.assertLess(fails, trials * 0.1)

  def test_von_mises_sample_average_gradient(self):
    loc = jnp.array([1.] * 7)
    concentration = np.logspace(-3, 3, 7)
    grad_ys = np.ones(7, dtype=np.float32)
    n = 1000

    def loss(loc, concentration):
      vm = self.distrax_cls(loc, concentration)
      samples = vm.sample(sample_shape=(n,), seed=1)
      return jnp.mean(samples, axis=0)

    grad_loc, grad_concentration = jnp.vectorize(
        jax.grad(loss, argnums=(0, 1)),
        signature='(),()->(),()',
    )(loc, concentration)

    # dsamples / dloc = 1 => dloss / dloc = dloss / dsamples = grad_ys
    np.testing.assert_allclose(grad_loc, grad_ys, atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(grad_concentration, [0.]*7, atol=1e-1, rtol=1e-1)

  def test_von_mises_sample_circular_variance_gradient(self):
    loc = jnp.array([1.] * 7)
    concentration = np.logspace(-3, 3, 7)
    n = 1000

    def loss(loc, concentration):
      vm = self.distrax_cls(loc, concentration)
      samples = vm.sample(sample_shape=(n,), seed=1)
      return jnp.mean(1-jnp.cos(samples-loc), axis=0)

    grad_loc, grad_concentration = jnp.vectorize(
        jax.grad(loss, argnums=(0, 1)),
        signature='(),()->(),()',
    )(loc, concentration)

    def analytical_loss(concentration):
      i1e = jax.scipy.special.i1e(concentration)
      i0e = jax.scipy.special.i0e(concentration)
      return 1. - i1e / i0e

    expected_grad_concentration = jnp.vectorize(
        jax.grad(analytical_loss)
    )(concentration)

    np.testing.assert_allclose(grad_loc, [0.] * 7, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(
        grad_concentration, expected_grad_concentration, atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
  absltest.main()
