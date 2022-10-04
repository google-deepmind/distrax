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
"""Tests for `multinomial.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import multinomial
from distrax._src.utils import equivalence
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


class MultinomialTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(multinomial.Multinomial)
    self.total_count = np.asarray(
        [4, 3], dtype=np.float32)  # float dtype required for TFP
    self.probs = 0.5 * np.asarray([0.1, 0.4, 0.2, 0.3])  # unnormalized
    self.logits = np.log(self.probs)

  @parameterized.named_parameters(
      ('from probs', False),
      ('from logits', True))
  def test_parameters(self, from_logits):
    if from_logits:
      dist_params = {'logits': self.logits, 'total_count': self.total_count}
    else:
      dist_params = {'probs': self.probs, 'total_count': self.total_count}
    dist = self.distrax_cls(**dist_params)
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        dist.logits, np.tile(math.normalize(logits=self.logits), (2, 1)))
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        dist.probs, np.tile(math.normalize(probs=self.probs), (2, 1)))

  @parameterized.named_parameters(
      ('probs and logits', {
          'total_count': 3, 'logits': [0.1, -0.2], 'probs': [0.6, 0.4]}),
      ('both probs and logits are None', {
          'total_count': 3, 'logits': None, 'probs': None}),
      ('logits are 0d', {'total_count': 3, 'logits': 3.}),
      ('probs are 0d', {'total_count': 3, 'probs': 1.}),
      ('logits have wrong dim', {'total_count': 3, 'logits': np.ones((4, 1))}),
      ('probs have wrong dim', {'total_count': 3, 'probs': np.ones((4, 1))}),
      ('bool dtype', {
          'total_count': 3, 'logits': [0.1, 0.], 'dtype': jnp.bool_}),
      ('complex64 dtype', {
          'total_count': 3, 'logits': [0.1, 0.], 'dtype': jnp.complex64}),
      ('complex128 dtype', {
          'total_count': 3, 'logits': [0.1, 0.], 'dtype': jnp.complex128}),
  )
  def test_raises_on_invalid_inputs(self, dist_params):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @parameterized.named_parameters(
      ('1d logits', {'logits': [0.0, 1.0, -0.5]}),
      ('1d probs', {'probs': [0.2, 0.5, 0.3]}),
      ('2d logits', {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]}),
      ('2d probs', {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]}),
  )
  def test_event_shape(self, dist_params):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist_params.update({'total_count': self.total_count})
    super()._test_event_shape((), dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape',
       {'logits': [0.0, 1.0, -0.5]},
       ()),
      ('1d probs, no shape',
       {'probs': [0.2, 0.5, 0.3]},
       ()),
      ('1d logits, int shape',
       {'logits': [0.0, 1.0, -0.5]},
       1),
      ('1d probs, int shape',
       {'probs': [0.2, 0.5, 0.3]},
       1),
      ('1d logits, 1-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (1,)),
      ('1d probs, 1-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (1,)),
      ('1d logits, 2-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (5, 4)),
      ('1d probs, 2-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (5, 4)),
      ('2d logits, no shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5, 4)),
  )
  def test_sample_shape_with_int_total_count(
      self, dist_params, sample_shape):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist_params.update({
        'total_count': 3,
    })
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=dist_params,
        sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape',
       {'logits': [0.0, 1.0, -0.5]},
       ()),
      ('1d probs, no shape',
       {'probs': [0.2, 0.5, 0.3]},
       ()),
      ('1d logits, int shape',
       {'logits': [0.0, 1.0, -0.5]},
       1),
      ('1d probs, int shape',
       {'probs': [0.2, 0.5, 0.3]},
       1),
      ('1d logits, 1-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (1,)),
      ('1d probs, 1-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (1,)),
      ('1d logits, 2-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (5, 4)),
      ('1d probs, 2-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (5, 4)),
      ('2d logits, no shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5, 4)),
  )
  def test_sample_shape_with_1d_total_count(
      self, dist_params, sample_shape):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist_params.update({
        'total_count': np.asarray([4, 3], dtype=np.float32),
    })
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=dist_params,
        sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape',
       {'logits': [0.0, 1.0, -0.5]},
       ()),
      ('1d probs, no shape',
       {'probs': [0.2, 0.5, 0.3]},
       ()),
      ('1d logits, int shape',
       {'logits': [0.0, 1.0, -0.5]},
       1),
      ('1d probs, int shape',
       {'probs': [0.2, 0.5, 0.3]},
       1),
      ('1d logits, 1-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (1,)),
      ('1d probs, 1-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (1,)),
      ('1d logits, 2-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (5, 4)),
      ('1d probs, 2-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (5, 4)),
      ('2d logits, no shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5, 4)),
  )
  def test_sample_shape_with_2d_total_count(
      self, dist_params, sample_shape):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    total_count = np.asarray(
        [[4, 3], [5, 4], [3, 2], [1, 4]], dtype=np.float32)
    dist_params.update({'total_count': total_count})
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=dist_params,
        sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape',
       {'logits': [0.0, 1.0, -0.5]},
       ()),
      ('1d probs, no shape',
       {'probs': [0.2, 0.5, 0.3]},
       ()),
      ('1d logits, int shape',
       {'logits': [0.0, 1.0, -0.5]},
       1),
      ('1d probs, int shape',
       {'probs': [0.2, 0.5, 0.3]},
       1),
      ('1d logits, 1-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (1,)),
      ('1d probs, 1-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (1,)),
      ('1d logits, 2-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (5, 4)),
      ('1d probs, 2-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (5, 4)),
      ('2d logits, no shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5, 4)),
  )
  def test_sum_samples(
      self, dist_params, sample_shape):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    total_count = np.asarray(
        [[4, 3], [5, 4], [3, 2], [1, 4]], dtype=np.float32)
    dist_params.update({'total_count': total_count})
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=sample_shape))
    samples = sample_fn(self.key)
    sum_samples = jnp.sum(samples, axis=-1)
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        np.asarray(sum_samples, dtype=np.float32),
        np.broadcast_to(total_count, sum_samples.shape))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape',
       {'logits': [0.0, 1.0, -0.5]},
       ()),
      ('1d probs, no shape',
       {'probs': [0.2, 0.5, 0.3]},
       ()),
      ('1d logits, int shape',
       {'logits': [0.0, 1.0, -0.5]},
       1),
      ('1d probs, int shape',
       {'probs': [0.2, 0.5, 0.3]},
       1),
      ('1d logits, 1-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (1,)),
      ('1d probs, 1-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (1,)),
      ('1d logits, 2-tuple shape',
       {'logits': [0.0, 1.0, -0.5]},
       (5, 4)),
      ('1d probs, 2-tuple shape',
       {'probs': [0.2, 0.5, 0.3]},
       (5, 4)),
      ('2d logits, no shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       (5, 4)),
  )
  def test_sample_and_log_prob(self, dist_params, sample_shape):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    total_count = np.asarray(
        [[4, 3], [5, 4], [3, 2], [1, 4]], dtype=np.float32)
    dist_params.update({'total_count': total_count})
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs=dist_params,
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(atol=1e-6, rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('int32', jnp.int32),
      ('int64', jnp.int64),
      ('uint32', jnp.uint32),
      ('uint64', jnp.uint64),
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist_params = {
        'logits': self.logits, 'dtype': dtype, 'total_count': self.total_count}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  def test_sample_extreme_probs(self):
    dist_params = {
        'probs': np.asarray([1., 0., 0., 0.]), 'total_count': 10}
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=100))
    samples = sample_fn(self.key)
    np.testing.assert_equal(np.unique(samples[..., 0]), 10)
    np.testing.assert_equal(np.unique(samples[..., 1:]), 0)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, 1 input',
       {'logits': [0.0, 0.5, -0.5]},
       [2, 1, 0]),
      ('1d logits, 2 inputs',
       {'logits': [0.0, 0.5, -0.5]},
       [[1, 2, 0], [0, 1, 2]]),
      ('2d logits, 2 inputs',
       {'logits': [[0.0, 0.5, -0.5], [-0.1, 0.1, 0.1]]},
       [[1, 0, 2], [1, 1, 1]]),
      ('2d logits, rank-3 inputs',
       {'logits': [[0.0, 0.5, -0.5], [-0.1, 0.1, 0.1]]},
       np.asarray([[1, 2, 0], [1, 0, 2]])[None, ...]),
      ('1d probs, 1 input',
       {'probs': [0.3, 0.2, 0.5]},
       [1, 2, 0]),
      ('1d probs, 2 inputs',
       {'probs': [0.3, 0.2, 0.5]},
       [[1, 0, 2], [1, 1, 1]]),
      ('2d probs, 2 inputs',
       {'probs': [[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]},
       [[1, 2, 0], [2, 1, 0]]),
      ('2d probs, rank-3 inputs',
       {'probs': [[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]},
       np.asarray([[1, 0, 2], [1, 1, 1]])[None, ...]),
  )
  def test_log_prob(self, dist_params, value):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist_params.update({'total_count': 3})
    value = jnp.asarray(value)
    super()._test_attribute(
        attribute_string='log_prob',
        dist_kwargs=dist_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(atol=1e-6, rtol=1e-3))

  @chex.all_variants(with_jit=False, with_pmap=False)
  def test_log_prob_extreme_probs(self):
    dist_params = {
        'probs': np.array([0.0, 1.0, 0.0]),
        'total_count': 3,
    }
    value = np.array([[0, 3, 0], [1, 1, 1]])
    expected_result = np.asarray([0., -np.inf])
    dist = self.distrax_cls(**dist_params)
    np.testing.assert_allclose(
        self.variant(dist.log_prob)(value), expected_result, atol=1e-5)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('from 2d logits',
       {'logits': np.asarray([[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]])}),
      ('from 2d probs',
       {'probs': np.asarray([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]])}),
  )
  def test_entropy(self, dist_params):
    # The TFP Multinomial does not implement `entropy`, so we use scipy for
    # the tests.
    dist_params.update({
        'total_count': np.asarray([3, 10]),
    })
    dist = self.distrax_cls(**dist_params)
    entropy = []
    for probs, counts in zip(dist.probs, dist.total_count):
      entropy.append(stats.multinomial(n=counts, p=probs).entropy())
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        self.variant(dist.entropy)(), np.asarray(entropy))

  @chex.all_variants(with_pmap=False)
  def test_entropy_extreme_probs(self):
    dist_params = {
        'probs': np.asarray([1.0, 0.0, 0.0]),
        'total_count': np.asarray([3, 10]),
    }
    dist = self.distrax_cls(**dist_params)
    expected_result = np.asarray([0., 0.])
    np.testing.assert_allclose(
        self.variant(dist.entropy)(), expected_result, atol=3e-4)

  @chex.all_variants(with_pmap=False)
  def test_entropy_scalar(self):
    # The TFP Multinomial does not implement `entropy`, so we use scipy for
    # the tests.
    probs = np.asarray([0.1, 0.5, 0.4])
    total_count = 5
    scipy_entropy = stats.multinomial(n=total_count, p=probs).entropy()
    distrax_entropy_fn = self.variant(
        lambda x, y: multinomial.Multinomial._entropy_scalar(total_count, x, y))
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        distrax_entropy_fn(probs, np.log(probs)), scipy_entropy)

  @chex.all_variants(with_pmap=False)
  def test_entropy_scalar_extreme_probs(self):
    probs = np.asarray([1., 0., 0.])
    total_count = 5
    expected_result = 0.
    distrax_entropy_fn = self.variant(
        lambda x, y: multinomial.Multinomial._entropy_scalar(total_count, x, y))
    np.testing.assert_allclose(
        distrax_entropy_fn(probs, np.log(probs)), expected_result, atol=1e-5)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('variance; from 2d logits',
       'variance', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
      ('variance; from 2d probs',
       'variance', {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
      ('mean; from 2d logits',
       'mean', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
      ('mean; from 2d probs',
       'mean', {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
      ('covariance; from 2d logits',
       'covariance', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
      ('covariance; from 2d probs',
       'covariance', {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
  )
  def test_method(self, function_string, dist_params):
    dist_params = {k: jnp.asarray(v) for k, v in dist_params.items()}
    total_count = np.asarray(
        [[4, 3], [5, 4], [3, 2], [1, 4]], dtype=np.float32)
    dist_params.update({'total_count': total_count})
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=dist_params,
        assertion_fn=self.assertion_fn(atol=1e-6, rtol=1e-3))

  def test_jittable(self):
    super()._test_jittable(
        dist_kwargs={
            'probs': np.asarray([1.0, 0.0, 0.0]),
            'total_count': np.asarray([3, 10])
        },
        assertion_fn=self.assertion_fn(atol=3e-4, rtol=1e-3))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
  )
  def test_slice(self, slice_):
    logits = jnp.array(np.random.randn(3, 4, 5))
    probs = jax.nn.softmax(jnp.array(np.random.randn(3, 4, 5)), axis=-1)
    total_count = jnp.full((3, 4), fill_value=2)
    dist1 = self.distrax_cls(total_count=total_count, logits=logits)
    dist2 = self.distrax_cls(total_count=total_count, probs=probs)
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        dist2[slice_].total_count, total_count[slice_])
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        jax.nn.softmax(dist1[slice_].logits, axis=-1),
        jax.nn.softmax(logits[slice_], axis=-1))
    self.assertion_fn(atol=1e-6, rtol=1e-3)(dist2[slice_].probs, probs[slice_])

  def test_slice_ellipsis(self):
    logits = jnp.array(np.random.randn(4, 4, 5))
    probs = jax.nn.softmax(jnp.array(np.random.randn(4, 4, 5)), axis=-1)
    total_count_value = 2
    total_count = jnp.full((4, 4), fill_value=total_count_value)
    dist1 = self.distrax_cls(total_count=total_count_value, logits=logits)
    dist2 = self.distrax_cls(total_count=total_count_value, probs=probs)
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        dist1[..., -1].total_count, total_count[..., -1])
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        dist2[..., -1].total_count, total_count[..., -1])
    self.assertion_fn(atol=1e-6, rtol=1e-3)(
        jax.nn.softmax(dist1[..., -1].logits, axis=-1),
        jax.nn.softmax(logits[:, -1], axis=-1))
    self.assertion_fn(atol=1e-6, rtol=1e-3)(dist2[..., -1].probs, probs[:, -1])


if __name__ == '__main__':
  absltest.main()
