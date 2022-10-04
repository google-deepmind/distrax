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
"""Tests for `laplace.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import laplace
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


class LaplaceTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(laplace.Laplace)

  @parameterized.named_parameters(
      ('1d std laplace', (0, 1)),
      ('2d std laplace', (np.zeros(2), np.ones(2))),
      ('rank 2 std normal', (np.zeros((3, 2)), np.ones((3, 2)))),
      ('broadcasted loc', (0, np.ones(3))),
      ('broadcasted scale', (np.ones(3), 1)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std laplace, no shape', (0, 1), ()),
      ('1d std laplace, int shape', (0, 1), 1),
      ('1d std laplace, 1-tuple shape', (0, 1), (1,)),
      ('1d std laplace, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std laplace, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std laplace, int shape', ([0, 0], [1, 1]), 1),
      ('2d std laplace, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std laplace, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
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
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(
        loc=jnp.zeros((), dtype), scale=jnp.ones((), dtype))
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std laplace, no shape', (0, 1), ()),
      ('1d std laplace, int shape', (0, 1), 1),
      ('1d std laplace, 1-tuple shape', (0, 1), (1,)),
      ('1d std laplace, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std laplace, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std laplace, int shape', ([0, 0], [1, 1]), 1),
      ('2d std laplace, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std laplace, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
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
        assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0.5, 0.1), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5, 0.1), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2), 0.3 * np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 0.8), 1),
      ('2d dist, 2d value', ([0.1, -0.5], 0.9 * np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), 200),
  )
  def test_log_prob(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='log_prob',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0.5, 0.1), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5, 0.1), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2), 0.3 * np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 0.8), 1),
      ('2d dist, 2d value', ([0.1, -0.5], 0.9 * np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), 200),
  )
  def test_prob(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='prob',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0.5, 0.1), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5, 0.1), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2), 0.3 * np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 0.8), 1),
      ('2d dist, 2d value', ([0.1, -0.5], 0.9 * np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), 200),
  )
  def test_cdf(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='cdf',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0.5, 0.1), np.array([1, 2])),
      ('1d dist, 2d value as list', (0.5, 0.1), [1, 2]),
      ('2d dist, 1d value', (0.5 + np.zeros(2), 0.3 * np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 0.8), 1),
      ('2d dist, 2d value', ([0.1, -0.5], 0.9 * np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), 200),
  )
  def test_log_cdf(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    super()._test_attribute(
        attribute_string='log_cdf',
        dist_args=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', ([0., 1., -0.5], [0.5, 1., 1.5]), 'entropy'),
      ('entropy broadcasted loc', (0.5, [0.5, 1., 1.5]), 'entropy'),
      ('entropy broadcasted scale', ([0., 1., -0.5], 0.8), 'entropy'),
      ('mean', ([0., 1., -0.5], [0.5, 1., 1.5]), 'mean'),
      ('mean broadcasted loc', (0.5, [0.5, 1., 1.5]), 'mean'),
      ('mean broadcasted scale', ([0., 1., -0.5], 0.8), 'mean'),
      ('variance', ([0., 1., -0.5], [0.5, 1., 1.5]), 'variance'),
      ('variance broadcasted loc', (0.5, [0.5, 1., 1.5]), 'variance'),
      ('variance broadcasted scale', ([0., 1., -0.5], 0.8), 'variance'),
      ('stddev', ([0., 1., -0.5], [0.5, 1., 1.5]), 'stddev'),
      ('stddev broadcasted loc', (0.5, [0.5, 1., 1.5]), 'stddev'),
      ('stddev broadcasted scale', ([0., 1., -0.5], 0.8), 'stddev'),
      ('mode', ([0., 1., -0.5], [0.5, 1., 1.5]), 'mode'),
      ('mode broadcasted loc', (0.5, [0.5, 1., 1.5]), 'mode'),
      ('mode broadcasted scale', ([0., 1., -0.5], 0.8), 'mode'),
  )
  def test_method(self, distr_params, function_string):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_attribute(
        attribute_string=function_string,
        dist_args=distr_params,
        assertion_fn=self.assertion_fn(rtol=2e-2))

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
    self.assertion_fn(rtol=2e-2)(self.variant(dist.median)(), dist.mean())

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
    rng = np.random.default_rng(42)
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'loc': rng.normal(size=(4, 1, 2)),
            'scale': np.array([[0.8, 0.2], [0.1, 1.2], [1.4, 3.1]]),
        },
        dist2_kwargs={
            'loc': rng.normal(size=(3, 2)),
            'scale': 0.1 + rng.uniform(size=(4, 1, 2)),
        },
        assertion_fn=self.assertion_fn(rtol=2e-2))

  def test_jitable(self):
    super()._test_jittable((0., 1.))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(42)
    loc = jnp.array(rng.normal(size=(3, 4, 5)))
    scale = jnp.array(rng.uniform(size=(3, 4, 5)))
    dist = self.distrax_cls(loc=loc, scale=scale)
    self.assertion_fn(rtol=2e-2)(dist[slice_].mean(), loc[slice_])

  def test_slice_different_parameterization(self):
    rng = np.random.default_rng(42)
    loc = jnp.array(rng.normal(size=(4,)))
    scale = jnp.array(rng.uniform(size=(3, 4)))
    dist = self.distrax_cls(loc=loc, scale=scale)
    self.assertion_fn(rtol=2e-2)(dist[0].loc, loc)  # Not slicing loc.
    self.assertion_fn(rtol=2e-2)(dist[0].scale, scale[0])


if __name__ == '__main__':
  absltest.main()
