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
"""Tests for `deterministic.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import deterministic
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


class DeterministicTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(deterministic.Deterministic)

  def test_loc(self):
    dist_params = {'loc': [0.1, 0.5, 1.5]}
    dist = self.distrax_cls(**dist_params)
    self.assertion_fn(rtol=1e-3)(dist.loc, dist_params['loc'])

  @parameterized.named_parameters(
      ('None', None),
      ('0.0', 0.0),
      ('0.1', 0.1))
  def test_atol(self, atol):
    dist_params = {'loc': np.asarray([0.1, 0.5, 1.5]), 'atol': atol}
    dist = self.distrax_cls(**dist_params)
    broadcasted_atol = np.zeros((3,)) if atol is None else atol * np.ones((3,))
    self.assertion_fn(rtol=1e-3)(dist.atol, broadcasted_atol)

  @parameterized.named_parameters(
      ('None', None),
      ('0.0', 0.0),
      ('0.1', 0.1))
  def test_rtol(self, rtol):
    dist_params = {'loc': np.asarray([0.1, 0.5, 1.5]), 'rtol': rtol}
    dist = self.distrax_cls(**dist_params)
    broadcasted_rtol = np.zeros((3,)) if rtol is None else rtol * np.ones((3,))
    self.assertion_fn(rtol=1e-3)(dist.rtol, broadcasted_rtol)

  @parameterized.named_parameters(
      ('atol_None_rtol_None', None, None),
      ('atol_0.1_rtol_None', 0.1, None),
      ('atol_None_rtol_0.1', None, 0.1),
      ('atol_0.05_rtol_0.1', 0.05, 0.1))
  def test_slack(self, atol, rtol):
    loc = np.asarray([0.1, 0.5, 1.5])
    target_value = (0 if atol is None else atol) + (
        0 if rtol is None else rtol) * np.abs(loc)
    dist_params = {'loc': loc, 'rtol': rtol, 'atol': atol}
    dist = self.distrax_cls(**dist_params)
    self.assertion_fn(rtol=1e-3)(dist.slack, target_value)

  def test_invalid_parameters(self):
    self._test_raises_error(
        dist_kwargs={'loc': 2., 'atol': np.array([0.1, 0.2])})
    self._test_raises_error(
        dist_kwargs={'loc': 2., 'rtol': np.array([0.1, 0.2])})

  @parameterized.named_parameters(
      ('1d', np.asarray([0., 1.])),
      ('2d', np.zeros((2, 3))),
  )
  def test_event_shape(self, loc):
    dist_params = {'loc': loc}
    super()._test_event_shape((), dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d, no shape', [0., 1.], ()),
      ('1d, int shape', [0., 1.], 1),
      ('1d, 1-tuple shape', [0., 1.], (1,)),
      ('1d, 2-tuple shape', [0., 1.], (2, 2)),
      ('2d, no shape', np.zeros((2, 3)), ()),
      ('2d, int shape', np.zeros((2, 3)), 1),
      ('2d, 1-tuple shape', np.zeros((2, 3)), (1,)),
      ('2d, 2-tuple shape', np.zeros((2, 3)), (5, 4)),
  )
  def test_sample_shape(self, loc, sample_shape):
    dist_params = {'loc': np.asarray(loc)}
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=dist_params,
        sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('int32', jnp.int32),
      ('int64', jnp.int64),
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(loc=jnp.zeros((), dtype=dtype))
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d, no shape', [0., 1.], ()),
      ('1d, int shape', [0., 1.], 1),
      ('1d, 1-tuple shape', [0., 1.], (1,)),
      ('1d, 2-tuple shape', [0., 1.], (2, 2)),
      ('2d, no shape', np.zeros((2, 3)), ()),
      ('2d, int shape', np.zeros((2, 3)), 1),
      ('2d, 1-tuple shape', np.zeros((2, 3)), (1,)),
      ('2d, 2-tuple shape', np.zeros((2, 3)), (5, 4)),
  )
  def test_sample_and_log_prob(self, loc, sample_shape):
    dist_params = {'loc': np.asarray(loc)}
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs=dist_params,
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob', 'log_prob'),
      ('prob', 'prob'),
      ('cdf', 'cdf'),
      ('log_cdf', 'log_cdf'),
  )
  def test_method_with_inputs_at_loc(self, function_string):
    loc = np.asarray([0.1, -0.9, 5.1])
    dist_params = {'loc': loc}
    inputs = np.repeat(loc[None, :], 10, axis=0)
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=dist_params,
        call_args=(inputs,),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob', 'log_prob'),
      ('prob', 'prob'),
      ('cdf', 'cdf'),
      ('log_cdf', 'log_cdf'),
  )
  def test_method_with_inputs_at_random_inputs(self, function_string):
    loc = np.asarray([0.1, -0.9, 5.1])
    dist_params = {'loc': loc}
    inputs = 0.1 * np.random.normal(size=(10,) + (len(loc),))
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=dist_params,
        call_args=(inputs,),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob_stddev0', 'log_prob', 0.0, 0.05, 0.1),
      ('log_prob_stddev0.05', 'log_prob', 0.05, 0.05, 0.1),
      ('log_prob_stddev0.1', 'log_prob', 0.1, 0.05, 0.1),
      ('prob_stddev0', 'prob', 0.0, 0.05, 0.1),
      ('prob_stddev0.05', 'prob', 0.05, 0.05, 0.1),
      ('prob_stddev0.1', 'prob', 0.1, 0.05, 0.1),
      ('cdf_stddev0', 'cdf', 0.0, 0.05, 0.1),
      ('cdf_stddev0.05', 'cdf', 0.05, 0.05, 0.1),
      ('cdf_stddev0.1', 'cdf', 0.1, 0.05, 0.1),
      ('log_cdf_stddev0', 'log_cdf', 0.0, 0.05, 0.1),
      ('log_cdf_stddev0.05', 'log_cdf', 0.05, 0.05, 0.1),
      ('log_cdf_stddev0.1', 'log_cdf', 0.1, 0.05, 0.1),
  )
  def test_method_with_inputs_and_slack(self, function_string, inputs_stddev,
                                        atol, rtol):
    loc = np.asarray([[4., -1., 0.], [0.5, 0.1, -8.]])
    dist_params = {'loc': loc, 'atol': atol, 'rtol': rtol}
    inputs = loc[None, ...] + inputs_stddev * np.random.normal(
        size=(20,) + loc.shape)
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=dist_params,
        call_args=(inputs,),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', [0., 1.], 'entropy'),
      ('mean', [0., 1.], 'mean'),
      ('mode', [0., 1.], 'mode'),
      ('variance', [0., 1.], 'variance'),
      ('variance from rank-2 params', np.ones((2, 3)), 'variance'),
      ('stddev', [-1.], 'stddev'),
      ('stddev from rank-2 params', -np.ones((2, 3)), 'stddev'),
  )
  def test_method(self, distr_params, function_string):
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs={'loc': np.asarray(distr_params)},
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_two_distributions(self, function_string, mode_string):
    loc1 = np.random.randn(3)
    loc2 = np.stack([loc1, np.random.randn(3)], axis=0)
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'loc': loc1,
        },
        dist2_kwargs={
            'loc': loc2,
        },
        assertion_fn=self.assertion_fn(rtol=1e-3))

  def test_jittable(self):
    super()._test_jittable((np.array([0., 4., -1., 4.]),))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    loc = jnp.array(np.random.randn(3, 4, 5))
    atol = jnp.array(np.random.randn(3, 4, 5))
    rtol = jnp.array(np.random.randn(3, 4, 5))
    dist = self.distrax_cls(loc=loc, atol=atol, rtol=rtol)
    self.assertion_fn(rtol=1e-3)(dist[slice_].loc, loc[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].atol, atol[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].rtol, rtol[slice_])

  def test_slice_different_parameterization(self):
    loc = jnp.array(np.random.randn(3, 4, 5))
    atol = jnp.array(np.random.randn(4, 5))
    rtol = jnp.array(np.random.randn(4, 5))
    dist = self.distrax_cls(loc=loc, atol=atol, rtol=rtol)
    self.assertion_fn(rtol=1e-3)(dist[0].loc, loc[0])
    self.assertion_fn(rtol=1e-3)(dist[0].atol, atol)  # Not slicing atol.
    self.assertion_fn(rtol=1e-3)(dist[0].rtol, rtol)  # Not slicing rtol.


if __name__ == '__main__':
  absltest.main()
