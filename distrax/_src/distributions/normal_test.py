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
"""Tests for `normal.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import normal
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np


class NormalTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(normal.Normal)

  @parameterized.named_parameters(
      ('1d std normal', (0, 1)),
      ('2d std normal', (np.zeros(2), np.ones(2))),
      ('rank 2 std normal', (np.zeros((3, 2)), np.ones((3, 2)))),
      ('broadcasted loc', (0, np.ones(3))),
      ('broadcasted scale', (np.ones(3), 1)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std normal, no shape', (0, 1), ()),
      ('1d std normal, int shape', (0, 1), 1),
      ('1d std normal, 1-tuple shape', (0, 1), (1,)),
      ('1d std normal, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std normal, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std normal, int shape', ([0, 0], [1, 1]), 1),
      ('2d std normal, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std normal, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
      ('rank 2 std normal, 2-tuple shape', (np.zeros((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted loc', (0, np.ones(3)), (2, 2)),
      ('broadcasted scale', (np.ones(3), 1), ()),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @jax.numpy_rank_promotion('raise')
  @parameterized.named_parameters(
      ('1d std normal, no shape', (0, 1), ()),
      ('1d std normal, int shape', (0, 1), 1),
      ('1d std normal, 1-tuple shape', (0, 1), (1,)),
      ('1d std normal, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std normal, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std normal, int shape', ([0, 0], [1, 1]), 1),
      ('2d std normal, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std normal, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
      ('rank 2 std normal, 2-tuple shape', (np.zeros((3, 2)), np.ones(
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
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0., 1.), np.array([1, 2])),
      ('2d dist, 1d value', (np.zeros(2), np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 1), 1),
      ('2d dist, 2d value', (np.zeros(2), np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), 200),
  )
  def test_method_with_input(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    for method in ['log_prob', 'prob', 'cdf', 'log_cdf', 'survival_function',
                   'log_survival_function']:
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
      ('no broadcast', ([0., 1., -0.5], [0.5, 1., 1.5])),
      ('broadcasted loc', (0.5, [0.5, 1., 1.5])),
      ('broadcasted scale', ([0., 1., -0.5], 0.8)),
  )
  def test_median(self, distr_params):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    dist = self.distrax_cls(*distr_params)
    self.assertion_fn(rtol=1e-2)(self.variant(dist.median)(), dist.mean())

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
            'loc': np.random.randn(4, 1, 2),
            'scale': np.asarray([[0.8, 0.2], [0.1, 1.2], [1.4, 3.1]]),
        },
        dist2_kwargs={
            'loc': np.random.randn(3, 2),
            'scale': 0.1 + np.random.rand(4, 1, 2),
        },
        assertion_fn=self.assertion_fn(rtol=3e-2))

  def test_jittable(self):
    super()._test_jittable((np.zeros((3,)), np.ones((3,))))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    loc = jnp.array(np.random.randn(3, 4, 5))
    scale = jnp.array(np.random.randn(3, 4, 5))
    dist = self.distrax_cls(loc=loc, scale=scale)
    self.assertion_fn(rtol=1e-2)(dist[slice_].mean(), loc[slice_])

  def test_slice_different_parameterization(self):
    loc = jnp.array(np.random.randn(4))
    scale = jnp.array(np.random.randn(3, 4))
    dist = self.distrax_cls(loc=loc, scale=scale)
    self.assertion_fn(rtol=1e-2)(dist[0].mean(), loc)  # Not slicing loc.
    self.assertion_fn(rtol=1e-2)(dist[0].stddev(), scale[0])

  def test_vmap_inputs(self):
    def log_prob_sum(dist, x):
      return dist.log_prob(x).sum()

    dist = normal.Normal(
        jnp.arange(3 * 4 * 5).reshape((3, 4, 5)), jnp.ones((3, 4, 5)))
    x = jnp.zeros((3, 4, 5))

    with self.subTest('no vmap'):
      actual = log_prob_sum(dist, x)
      expected = dist.log_prob(x).sum()
      self.assertion_fn()(actual, expected)

    with self.subTest('axis=0'):
      actual = jax.vmap(log_prob_sum, in_axes=0)(dist, x)
      expected = dist.log_prob(x).sum(axis=(1, 2))
      self.assertion_fn()(actual, expected)

    with self.subTest('axis=1'):
      actual = jax.vmap(log_prob_sum, in_axes=1)(dist, x)
      expected = dist.log_prob(x).sum(axis=(0, 2))
      self.assertion_fn()(actual, expected)

  def test_vmap_outputs(self):
    def summed_dist(loc, scale):
      return normal.Normal(loc.sum(keepdims=True), scale.sum(keepdims=True))

    loc = jnp.arange((3 * 4 * 5)).reshape((3, 4, 5))
    scale = jnp.ones((3, 4, 5))

    actual = jax.vmap(summed_dist)(loc, scale)
    expected = normal.Normal(
        loc.sum(axis=(1, 2), keepdims=True),
        scale.sum(axis=(1, 2), keepdims=True))

    np.testing.assert_equal(actual.batch_shape, expected.batch_shape)
    np.testing.assert_equal(actual.event_shape, expected.event_shape)

    x = jnp.array([[[1]], [[2]], [[3]]])
    self.assertion_fn(rtol=1e-6)(actual.log_prob(x), expected.log_prob(x))


if __name__ == '__main__':
  absltest.main()
