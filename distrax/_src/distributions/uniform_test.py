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
"""Tests for `uniform.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import uniform
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np


class UniformTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(uniform.Uniform)

  @parameterized.named_parameters(
      ('1d', (0., 1.)),
      ('2d', (np.zeros(2), np.ones(2))),
      ('rank 2', (np.zeros((3, 2)), np.ones((3, 2)))),
      ('broadcasted low', (0., np.ones(3))),
      ('broadcasted high', (np.ones(3), 1.)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d, no shape', (0., 1.), ()),
      ('1d, int shape', (0., 1.), 1),
      ('1d, 1-tuple shape', (0., 1.), (1,)),
      ('1d, 2-tuple shape', (0., 1.), (2, 2)),
      ('2d, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d, int shape', (np.zeros(2), np.ones(2)), 1),
      ('2d, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d, 2-tuple shape', (np.zeros(2), np.ones(2)), (2, 2)),
      ('rank 2, 2-tuple shape', (np.zeros((3, 2)), np.ones((3, 2))), (2, 2)),
      ('broadcasted low', (0., np.ones(3)), (2, 2)),
      ('broadcasted high', (np.ones(3), 1.), ()),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @jax.numpy_rank_promotion('raise')
  @parameterized.named_parameters(
      ('1d, no shape', (0., 1.), ()),
      ('1d, int shape', (0., 1.), 1),
      ('1d, 1-tuple shape', (0., 1.), (1,)),
      ('1d, 2-tuple shape', (0., 1.), (2, 2)),
      ('2d, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d, int shape', (np.zeros(2), np.ones(2)), 1),
      ('2d, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d, 2-tuple shape', (np.zeros(2), np.ones(2)), (2, 2)),
      ('rank 2, 2-tuple shape', (np.zeros((3, 2)), np.ones((3, 2))), (2, 2)),
      ('broadcasted low', (0., np.ones(3)), (2, 2)),
      ('broadcasted high', (np.ones(3), 1.), ()),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    super()._test_sample_and_log_prob(
        dist_args=distr_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob', 'log_prob'),
      ('prob', 'prob'),
      ('cdf', 'cdf'),
      ('survival_function', 'survival_function'),
      ('log_survival_function', 'log_survival_function')
  )
  def test_method_with_inputs(self, function_string):
    inputs = 10. * np.random.normal(size=(100,))
    super()._test_attribute(
        function_string, dist_args=(-1, 1), call_args=(inputs,))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', (0., 1.), 'entropy'),
      ('mean', (0, 1), 'mean'),
      ('variance', (0, 1), 'variance'),
      ('variance from 1d params', (np.ones(2), np.ones(2)), 'mean'),
      ('stddev', (0, 1), 'stddev'),
      ('stddev from rank 2 params', (np.ones((2, 3)), np.ones(
          (2, 3))), 'stddev'),
  )
  def test_method(self, distr_params, function_string):
    super()._test_attribute(function_string, distr_params)

  @parameterized.named_parameters(
      ('low', 'low'),
      ('high', 'high'),
  )
  def test_attribute(self, attribute_string):
    super()._test_attribute(attribute_string)

  @chex.all_variants(with_pmap=False)
  def test_median(self):
    np.testing.assert_allclose(
        self.variant(self.distrax_cls(-1, 1).median)(), 0)

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
            'low': -0.5 + np.random.rand(4, 1, 2),
            'high': np.array([[1.8, 1.5], [1.1, 1.2], [1.4, 1.1]]),
        },
        dist2_kwargs={
            'low': -1.0 + np.random.rand(3, 2),
            'high': 1.5 + np.random.rand(4, 1, 2),
        },
        assertion_fn=self.assertion_fn(rtol=1e-3))

  def test_jittable(self):
    super()._test_jittable((0.0, 1.0))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    low = jnp.zeros((3, 4, 5))
    high = jnp.ones((3, 4, 5))
    dist = self.distrax_cls(low=low, high=high)
    self.assertion_fn(rtol=1e-3)(dist[slice_].low, low[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].high, high[slice_])

  def test_slice_different_parameterization(self):
    low = jnp.zeros((3, 4, 5))
    high = 1.
    dist = self.distrax_cls(low=low, high=high)
    self.assertion_fn(rtol=1e-3)(dist[..., -1].low, low[..., -1])
    self.assertEqual(dist[..., -1].high.shape, (3, 4))
    self.assertion_fn(rtol=1e-3)(dist[..., -1].high, high)  # Not slicing high.

if __name__ == '__main__':
  absltest.main()
