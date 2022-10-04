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
"""Tests for `quantized.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import independent
from distrax._src.distributions import quantized
from distrax._src.distributions import uniform
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class QuantizedTFPUniform(equivalence.EquivalenceTest):
  """Class to test distrax quantized distribution against TFP.

  The quantized distribution takes a distribution as input. These can be either
  distrax or TFP distributions. In this test, we set the base distribution to
  be a TFP uniform one. Subsequent tests evaluate other combinations.
  """

  def _make_tfp_base_distribution(self):
    return tfd.Uniform(0., 100.)

  _make_distrax_base_distribution = _make_tfp_base_distribution

  def setUp(self):
    super().setUp()
    self._init_distr_cls(quantized.Quantized)
    self.tfd_base_distribution = self._make_tfp_base_distribution()
    self.distrax_base_distribution = self._make_distrax_base_distribution()

  def test_event_shape(self):
    super()._test_event_shape((self.distrax_base_distribution,),
                              dict(),
                              tfp_dist_args=(self.tfd_base_distribution,))

  def test_batch_shape(self):
    super()._test_batch_shape((self.distrax_base_distribution,),
                              dict(),
                              tfp_dist_args=(self.tfd_base_distribution,))

  def test_low_and_high(self):
    distr_params = (uniform.Uniform(0., 100.), 0.5, 90.5)
    dist = self.distrax_cls(*distr_params)
    self.assertion_fn(rtol=1e-2)(dist.low, 1.)
    self.assertion_fn(rtol=1e-2)(dist.high, 90.)

  @chex.all_variants
  @parameterized.named_parameters(
      ('empty shape', ()),
      ('int shape', 10),
      ('2-tuple shape', (10, 20)),
  )
  def test_sample_shape(self, sample_shape):
    super()._test_sample_shape((self.distrax_base_distribution,),
                               dict(),
                               tfp_dist_args=(self.tfd_base_distribution,),
                               sample_shape=sample_shape)

  @chex.all_variants
  def test_sample_dtype(self):
    dist = self.distrax_cls(self.distrax_base_distribution)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(dist.dtype, samples.dtype)
    self.assertEqual(dist.dtype, self.distrax_base_distribution.dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs, no shape', (None, None), ()),
      ('noop cutoffs, no shape', (-10., 200.), ()),
      ('low cutoff, no shape', (10., None), ()),
      ('high cutoff, no shape', (None, 50.), ()),
      ('both cutoffs, no shape', (10., 50.), ()),
      ('both cutoffs, int shape', (10., 50.), 5),
      ('no cutoffs, 2-tuple shape', (None, None), (5, 4)),
      ('noop cutoffs, 2-tuple shape', (-10., 200.), (5, 4)),
      ('low cutoff, 2-tuple shape', (10., None), (5, 4)),
      ('high cutoff, 2-tuple shape', (None, 50.), (5, 4)),
      ('both cutoff, 2-tuple shape', (10., 50.), (5, 4)),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    super()._test_sample_and_log_prob(
        dist_args=(self.distrax_base_distribution,) + distr_params,
        dist_kwargs=dict(),
        tfp_dist_args=(self.tfd_base_distribution,) + distr_params,
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs, integer', (None, None), 20),
      ('noop cutoffs, integer', (-10., 200.), 20),
      ('low cutoff, integer', (10., None), 20),
      ('high cutoff, integer', (None, 50.), 20),
      ('both cutoffs, integer', (10., 50.), 20),
      ('both cutoffs, integer greater than cutoff', (10., 50.), 70),
      ('both cutoffs, integer smaller than cutoff', (10., 50.), 5),
      ('both cutoffs, 1-d array', (10., 50.), np.array([20, 30])),
      ('no cutoffs, 1-d array', (None, None), np.array([20, 30])),
      ('noop cutoffs, 1-d array', (-10., 200.), np.array([20, 30])),
      ('low cutoffs, 1-d array', (10., None), np.array([20, 30])),
      ('high cutoffs, 1-d array', (None, 50.), np.array([20, 30])),
  )
  def test_method_with_value(self, distr_params, value):
    for method in ['log_cdf', 'cdf', 'prob', 'survival_function',
                   'log_survival_function']:
      with self.subTest(method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=(self.distrax_base_distribution,) + distr_params,
            tfp_dist_args=(self.tfd_base_distribution,) + distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs, integer', (None, None), 20),
      ('noop cutoffs, integer', (-10., 200.), 20),
      ('low cutoff, integer', (10., None), 20),
      ('high cutoff, integer', (None, 50.), 20),
      ('both cutoffs, integer', (10., 50.), 20),
      ('both cutoffs, 1-d array', (10., 50.), np.array([20, 30])),
      ('no cutoffs, 1-d array', (None, None), np.array([20, 30])),
      ('noop cutoffs, 1-d array', (-10., 200.), np.array([20, 30])),
      ('low cutoffs, 1-d array', (10., None), np.array([20, 30])),
      ('high cutoffs, 1-d array', (None, 50.), np.array([20, 30])),
  )
  def test_log_prob(self, distr_params, value):
    """Tests the `log_prob`.

    We separate this test from `test_method_with_value` because the options
    where `value` is outside the cutoff return `nan` in TFP but `-inf` in
    Distrax.

    Args:
      distr_params: Parameters of the distribution.
      value: The value where the `log_prob` is evaluated.
    """
    super()._test_attribute(
        attribute_string='log_prob',
        dist_args=(self.distrax_base_distribution,) + distr_params,
        tfp_dist_args=(self.tfd_base_distribution,) + distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('below low', 10., 90., 8),
      ('above high', 10., 90., 95),
      ('below support', None, None, -10),
      ('above support', None, None, 101),
      ('within support, non-integer', None, None, 40.5),
  )
  def test_edge_cases(self, low, high, value):
    distr_params = {
        'distribution': self.distrax_base_distribution,
        'low': low,
        'high': high,
    }
    dist = self.distrax_cls(**distr_params)
    np.testing.assert_allclose(self.variant(dist.log_prob)(value), -np.inf)
    np.testing.assert_allclose(self.variant(dist.prob)(value), 0.)

  @parameterized.named_parameters(
      ('low with cutoffs', (10., 50.), 'low'),
      ('high with cutoffs', (10., 50.), 'high'),
  )
  def test_method(self, distr_params, function_string):
    super()._test_attribute(
        attribute_string=function_string,
        dist_args=(self.distrax_base_distribution,) + distr_params,
        dist_kwargs=dict(),
        tfp_dist_args=(self.tfd_base_distribution,) + distr_params,
        assertion_fn=self.assertion_fn(rtol=1e-2))

  def test_jittable(self):
    super()._test_jittable((self.tfd_base_distribution, 0., 1.))


class QuantizedDistraxUniform(QuantizedTFPUniform):

  def _make_distrax_base_distribution(self):
    return uniform.Uniform(0., 100.)

  def test_jittable(self):
    super()._test_jittable((self.distrax_base_distribution, 0., 1.))


class QuantizedTFPUniform2D(equivalence.EquivalenceTest):
  """Class to test distrax quantized distribution against TFP.

  The quantized distribution takes a distribution as input. These can be either
  distrax or TFP distributions. In this test, we set the base distribution to
  be a TFP uniform one with `batch_shape == (2,)`. Subsequent tests evaluate
  other combinations.
  """

  def _make_tfp_base_distribution(self):
    return tfd.Uniform(low=[0., 10.], high=[100., 90.])

  _make_distrax_base_distribution = _make_tfp_base_distribution

  def setUp(self):
    super().setUp()
    self._init_distr_cls(quantized.Quantized)
    self.tfd_base_distribution = self._make_tfp_base_distribution()
    self.distrax_base_distribution = self._make_distrax_base_distribution()

  def test_event_shape(self):
    kwargs = {
        'low': np.array([10., 30.], dtype=np.float32),
        'high': np.array([80., 70.], dtype=np.float32),
    }
    super()._test_event_shape(dist_args=(self.distrax_base_distribution,),
                              dist_kwargs=kwargs,
                              tfp_dist_args=(self.tfd_base_distribution,))

  def test_batch_shape(self):
    kwargs = {
        'low': np.array([10., 30.], dtype=np.float32),
        'high': np.array([80., 70.], dtype=np.float32),
    }
    super()._test_batch_shape(dist_args=(self.distrax_base_distribution,),
                              dist_kwargs=kwargs,
                              tfp_dist_args=(self.tfd_base_distribution,))

  @chex.all_variants
  @parameterized.named_parameters(
      ('empty shape', ()),
      ('int shape', 10),
      ('2-tuple shape', (10, 20)),
  )
  def test_sample_shape(self, sample_shape):
    kwargs = {
        'low': np.array([10., 30.], dtype=np.float32),
        'high': np.array([80., 70.], dtype=np.float32),
    }
    super()._test_sample_shape(dist_args=(self.distrax_base_distribution,),
                               dist_kwargs=kwargs,
                               tfp_dist_args=(self.tfd_base_distribution,),
                               sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs, no shape', (None, None), ()),
      ('scalar cutoffs, no shape', (20., 50.), ()),
      ('1d cutoffs, no shape', ([10., 10.], [20., 80.]), ()),
      ('mixed cutoffs, no shape', ([10., 10.], 20.), ()),
      ('no cutoffs, int shape', (None, None), 10),
      ('scalar cutoffs, int shape', (20., 50.), 10),
      ('1d cutoffs, int shape', ([10., 10.], [20., 80.]), 10),
      ('mixed cutoffs, int shape', ([10., 10.], 20.), 10),
      ('no cutoffs, 1d shape', (None, None), [10]),
      ('scalar cutoffs, 1d shape', (20., 50.), [10]),
      ('1d cutoffs, 1d shape', ([10., 10.], [20., 80.]), [10]),
      ('mixed cutoffs, 1d shape', ([10., 10.], 20.), [10]),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = tuple(map(
        lambda x: None if x is None else np.asarray(x, np.float32),
        distr_params))
    super()._test_sample_and_log_prob(
        dist_args=(self.distrax_base_distribution,) + distr_params,
        dist_kwargs=dict(),
        tfp_dist_args=(self.tfd_base_distribution,) + distr_params,
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs, scalar value', (None, None), 20),
      ('scalar cutoffs, scalar value', (20., 50.), 20),
      ('1d cutoffs, scalar value', ([10., 12.], [20., 80.]), 15),
      ('mixed cutoffs, scalar value', ([10., 15.], 20.), 18),
      ('no cutoffs, 1d value', (None, None), np.array([20, 30])),
      ('scalar cutoffs, 1d value', (20., 50.), np.array([20, 30])),
      ('1d cutoffs, 1d value', ([10., 20.], [20., 80.]), np.array([20, 20])),
      ('mixed cutoffs, 1d value', ([10., 15.], 20.), np.array([11, 20])),
      ('mixed cutoffs, 2d value',
       ([10., 15.], 80.), np.array([[15, 15], [10, 20], [15, 15]])),
  )
  def test_method_with_value(self, distr_params, value):
    # For `prob`, `log_prob`, `survival_function`, and `log_survival_function`
    # distrax and TFP agree on integer values. We do not test equivalence on
    # non-integer values where they may disagree.
    # We also do not test equivalence on values outside of the cutoff, where
    # `log_prob` values can be different (`NaN` vs. `-jnp.inf`).
    distr_params = tuple(map(
        lambda x: None if x is None else np.asarray(x, np.float32),
        distr_params))
    for method in ['log_cdf', 'cdf', 'prob', 'log_prob', 'survival_function',
                   'log_survival_function']:
      with self.subTest(method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=(self.distrax_base_distribution,) + distr_params,
            tfp_dist_args=(self.tfd_base_distribution,) + distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  def test_jittable(self):
    super()._test_jittable((self.tfd_base_distribution, 0., 1.))


class QuantizedDistraxUniform2D(QuantizedTFPUniform2D):

  def _make_distrax_base_distribution(self):
    return uniform.Uniform(low=[0., 10.], high=[100., 90.])

  def test_jittable(self):
    super()._test_jittable((self.distrax_base_distribution, 0., 1.))


class QuantizedInvalidParams(equivalence.EquivalenceTest):
  """Class to test invalid combinations of the input parameters."""

  def setUp(self):
    super().setUp()
    self._init_distr_cls(quantized.Quantized)

  def test_non_univariate(self):
    self._test_raises_error(dist_kwargs={
        'distribution': independent.Independent(
            uniform.Uniform(np.array([0., 0.]), np.array([1., 1,])),
            reinterpreted_batch_ndims=1),
    })

  def test_low_shape(self):
    self._test_raises_error(dist_kwargs={
        'distribution': uniform.Uniform(0., 1.),
        'low': np.zeros((4,))
    })

  def test_high_shape(self):
    self._test_raises_error(dist_kwargs={
        'distribution': uniform.Uniform(0., 1.),
        'high': np.ones((4,))
    })


class QuantizedSlicingTest(parameterized.TestCase):
  """Class to test the `getitem` method."""

  def setUp(self):
    super().setUp()
    self.uniform_low = np.random.randn(2, 3, 4)
    self.uniform_high = self.uniform_low + np.abs(np.random.randn(2, 3, 4))
    self.base = uniform.Uniform(self.uniform_low, self.uniform_high)
    self.low = np.ceil(np.random.randn(3, 4))
    self.high = np.floor(np.random.randn(3, 4))
    self.dist = quantized.Quantized(self.base, self.low, self.high)

  def assertion_fn(self, rtol):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  @parameterized.named_parameters(
      ('single element', 1, (3, 4)),
      ('range', slice(-1), (1, 3, 4)),
      ('range_2', (slice(None), slice(-1)), (2, 2, 4)),
      ('ellipsis', (Ellipsis, -1), (2, 3)),
  )
  def test_slice(self, slice_, expected_batch_shape):
    sliced_dist = self.dist[slice_]
    self.assertEqual(sliced_dist.batch_shape, expected_batch_shape)
    self.assertEqual(sliced_dist.event_shape, self.dist.event_shape)
    self.assertIsInstance(sliced_dist, quantized.Quantized)
    self.assertIsInstance(sliced_dist.distribution, self.base.__class__)
    self.assertion_fn(rtol=1e-2)(
        sliced_dist.distribution.low, self.uniform_low[slice_])
    self.assertion_fn(rtol=1e-2)(
        sliced_dist.distribution.high, self.uniform_high[slice_])
    self.assertion_fn(rtol=1e-2)(
        sliced_dist.low,
        np.broadcast_to(self.low, self.base.batch_shape)[slice_])
    self.assertion_fn(rtol=1e-2)(
        sliced_dist.high,
        np.broadcast_to(self.high, self.base.batch_shape)[slice_])


class QuantizedSurvivalFunctionConsistencyTest(parameterized.TestCase):
  """Class to test whether `survival_function` = `1. - cdf`.

  Test evaluates on both integer values and non-integer values.
  """

  def setUp(self):
    super().setUp()
    self.base_distribution = uniform.Uniform(0., 10.)
    self.values = np.linspace(-2., 12, num=57)  # -2, -1.75, -1.5, ..., 12.

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs', (None, None)),
      ('noop cutoffs', (-10., 20.)),
      ('low cutoff', (1., None)),
      ('high cutoff', (None, 5.)),
      ('both cutoffs', (1., 5.)),
  )
  def test_survival_function_cdf_consistency(self, dist_params):
    dist = quantized.Quantized(self.base_distribution, *dist_params)
    results = self.variant(
        lambda x: dist.cdf(x) + dist.survival_function(x))(self.values)
    np.testing.assert_allclose(results, np.ones_like(self.values), rtol=1e-2)

  @chex.all_variants
  @parameterized.named_parameters(
      ('no cutoffs', (None, None)),
      ('noop cutoffs', (-10., 20.)),
      ('low cutoff', (1., None)),
      ('high cutoff', (None, 5.)),
      ('both cutoffs', (1., 5.)),
  )
  def test_log_survival_function_log_cdf_consistency(self, dist_params):
    def _sum_exps(dist, x):
      return jnp.exp(dist.log_cdf(x)) + jnp.exp(dist.log_survival_function(x))
    dist = quantized.Quantized(self.base_distribution, *dist_params)
    results = self.variant(_sum_exps)(dist, self.values)
    np.testing.assert_allclose(results, np.ones_like(self.values), rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
