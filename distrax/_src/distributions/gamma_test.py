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
"""Tests for `gamma.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import gamma
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


class GammaTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(gamma.Gamma)

  @parameterized.named_parameters(
      ('0d params', (), (), ()),
      ('1d params', (2,), (2,), (2,)),
      ('2d params, no broadcast', (3, 2), (3, 2), (3, 2)),
      ('2d params, broadcasted concentration', (2,), (3, 2), (3, 2)),
      ('2d params, broadcasted rate', (3, 2), (2,), (3, 2)),
  )
  def test_properties(self, concentration_shape, rate_shape, batch_shape):
    rng = np.random.default_rng(42)
    concentration = 0.1 + rng.uniform(size=concentration_shape)
    rate = 0.1 + rng.uniform(size=rate_shape)
    dist = gamma.Gamma(concentration, rate)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=2e-2)(
        dist.concentration, np.broadcast_to(concentration, batch_shape))
    self.assertion_fn(rtol=2e-2)(dist.rate, np.broadcast_to(rate, batch_shape))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std gamma, no shape', (1, 1), ()),
      ('1d std gamma, int shape', (1, 1), 1),
      ('1d std gamma, 1-tuple shape', (1, 1), (1,)),
      ('1d std gamma, 2-tuple shape', (1, 1), (2, 2)),
      ('2d std gamma, no shape', (np.ones(2), np.ones(2)), ()),
      ('2d std gamma, int shape', ([1, 1], [1, 1]), 1),
      ('2d std gamma, 1-tuple shape', (np.ones(2), np.ones(2)), (1,)),
      ('2d std gamma, 2-tuple shape', ([1, 1], [1, 1]), (2, 2)),
      ('rank 2 std gamma, 2-tuple shape', (np.ones((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted concentration', (1, np.ones(3)), (2, 2)),
      ('broadcasted rate', (np.ones(3), 1), ()),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(
        concentration=jnp.ones((), dtype), rate=jnp.ones((), dtype))
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std gamma, no shape', (11, 3), ()),
      ('1d std gamma, int shape', (11, 3), 1),
      ('1d std gamma, 1-tuple shape', (11, 3), (1,)),
      ('1d std gamma, 2-tuple shape', (1, 1), (2, 2)),
      ('2d std gamma, no shape', (np.ones(2), np.ones(2)), ()),
      ('2d std gamma, int shape', ([1, 1], [1, 1]), 1),
      ('2d std gamma, 1-tuple shape', (np.ones(2), np.ones(2)), (1,)),
      ('2d std gamma, 2-tuple shape', ([1, 1], [1, 1]), (2, 2)),
      ('rank 2 std gamma, 2-tuple shape', (np.ones((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted concentration', (1, np.ones(3)), (2, 2)),
      ('broadcasted rate', (np.ones(3), 1), ()),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_and_log_prob(
        dist_args=distr_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (3.1, 1), 1),
      ('1d dist, 2d value', (0.5, 0.1), np.array([1, 2])),
      ('2d dist, 1d value', (0.5 + np.zeros(2), 0.3 * np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (0.4 + np.zeros(2), 0.8), 1),
      ('2d dist, 2d value', ([0.1, 0.5], 0.9 * np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (2.1, 1), 200),
  )
  def test_method_with_value(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    for method in ['log_prob', 'prob', 'cdf', 'log_cdf']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('no broadcast', ([0.1, 1.3, 0.5], [0.5, 1.3, 1.5])),
      ('broadcasted concentration', (0.5, [0.5, 1.3, 1.5])),
      ('broadcasted rate', ([0.1, 1.3, 0.5], 0.8)),
  )
  def test_method(self, distr_params):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    for method in ['entropy', 'mean', 'variance', 'stddev', 'mode']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=distr_params,
            assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax')
  )
  def test_with_two_distributions(self, function_string, mode_string):
    rtol = 1e-3
    atol = 1e-4
    rng = np.random.default_rng(42)
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'concentration': np.abs(
                rng.normal(size=(4, 1, 2))).astype(np.float32),
            'rate': np.array(
                [[0.8, 0.2], [0.1, 1.2], [1.4, 3.1]], dtype=np.float32),
        },
        dist2_kwargs={
            'concentration': np.abs(rng.normal(size=(3, 2))).astype(np.float32),
            'rate': 0.1 + rng.uniform(size=(4, 1, 2)).astype(np.float32),
        },
        assertion_fn=lambda x, y: np.testing.assert_allclose(x, y, rtol, atol))

  def test_jitable(self):
    super()._test_jittable(
        (0.1, 1.5), assertion_fn=self.assertion_fn(rtol=2e-2))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    concentration = jnp.array(np.abs(np.random.randn(3, 4, 5)))
    rate = jnp.array(np.abs(np.random.randn(3, 4, 5)))
    dist = self.distrax_cls(concentration, rate)
    self.assertion_fn(rtol=2e-2)(
        dist[slice_].concentration, concentration[slice_])
    self.assertion_fn(rtol=2e-2)(dist[slice_].rate, rate[slice_])

  def test_slice_different_parameterization(self):
    concentration = jnp.array(np.abs(np.random.randn(3, 4, 5)))
    rate = jnp.array(np.abs(np.random.randn(4, 5)))
    dist = self.distrax_cls(concentration, rate)
    self.assertion_fn(rtol=2e-2)(dist[0].concentration, concentration[0])
    self.assertion_fn(rtol=2e-2)(dist[0].rate, rate)  # Not slicing rate.


if __name__ == '__main__':
  absltest.main()
