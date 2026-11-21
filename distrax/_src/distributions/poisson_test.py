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
"""Tests for `poisson.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions import poisson
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


class PoissonTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(poisson.Poisson)

  @parameterized.named_parameters(
      ('1d poisson', (1,)),
      ('2d poisson', (np.ones(2),)),
      ('rank 2 poisson', (np.zeros((3, 2)),)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d poisson, no shape', (1,), ()),
      ('1d poisson, int shape', (1,), 1),
      ('1d poisson, 1-tuple shape', (1,), (1,)),
      ('1d poisson, 2-tuple shape', (1,), (2, 2)),
      ('2d poisson, no shape', (np.ones(2),), ()),
      ('2d poisson, int shape', ([1, 1],), 1),
      ('2d poisson, 1-tuple shape', (np.ones(2),), (1,)),
      ('2d poisson, 2-tuple shape', ([1, 1],), (2, 2)),
      (
          'rank 2 poisson, 2-tuple shape',
          (np.ones((3, 2)),),
          (2, 2),
      ),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32), ('float64', jnp.float64)
  )
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(loc=jnp.zeros((), dtype), scale=jnp.ones((), dtype))
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d poisson, no shape', (1,), ()),
      ('1d poisson, int shape', (1,), 1),
      ('1d poisson, 1-tuple shape', (1,), (1,)),
      ('1d poisson, 2-tuple shape', (1,), (2, 2)),
      ('2d poisson, no shape', (np.ones(2),), ()),
      ('2d poisson, int shape', ([1, 1],), 1),
      ('2d poisson, 1-tuple shape', (np.ones(2),), (1,)),
      ('2d poisson, 2-tuple shape', ([1, 1],), (2, 2)),
      (
          'rank 2 poisson, 2-tuple shape',
          (np.ones((3, 2)),),
          (2, 2),
      ),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    super()._test_sample_and_log_prob(
        dist_args=distr_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (1,), 1),
      ('1d dist, 2d value', (0.5,), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5,), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2),), 1),
      ('2d dist, 2d value', ([0.1, 0.5],), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (1,), 200),
  )
  def test_log_prob(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='log_prob',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (1,), 1),
      ('1d dist, 2d value', (0.5,), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5,), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2),), 1),
      ('2d dist, 2d value', ([0.1, 0.5],), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (1,), 200),
  )
  def test_prob(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='prob',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (1,), 1),
      ('1d dist, 2d value', (0.5,), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5,), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2),), 1),
      ('2d dist, 2d value', ([0.1, 0.5],), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (1,), 200),
  )
  def test_cdf(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='cdf',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (1,), 1),
      ('1d dist, 2d value', (0.5,), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5,), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2),), 1),
      ('2d dist, 2d value', ([0.1, 0.5],), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (1,), 200),
  )
  def test_log_cdf(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='log_cdf',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (1,), 1),
      ('1d dist, 2d value', (0.5,), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5,), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2),), 1),
      ('2d dist, 2d value', ([0.1, 0.5],), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (1,), 200),
  )
  def test_log_survival_function(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='log_survival_function',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('mean', ([0.1, 1.0, 0.5],), 'mean'),
      ('variance', ([0.1, 1.0, 0.5],), 'variance'),
      ('stddev', ([0.1, 1.0, 0.5],), 'stddev'),
      ('mode', ([0.1, 1.0, 0.5],), 'mode'),
  )
  def test_method(self, distr_params, function_string):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),)
    super()._test_attribute(
        attribute_string=function_string,
        dist_args=distr_params,
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'),
  )
  def test_with_two_distributions(self, function_string, mode_string):
    rng = np.random.default_rng(42)
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'rate': jnp.exp(rng.normal(size=(4, 1, 2))),
        },
        dist2_kwargs={
            'rate': jnp.exp(rng.normal(size=(3, 2))),
        },
        assertion_fn=self.assertion_fn(rtol=2e-2),
    )

  def test_jitable(self):
    super()._test_jittable((1.0,))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(42)
    rate = jnp.exp(jnp.array(rng.normal(size=(3, 4, 5))))
    dist = self.distrax_cls(rate=rate)
    self.assertion_fn(rtol=2e-2)(dist[slice_].mean(), rate[slice_])


if __name__ == '__main__':
  absltest.main()
