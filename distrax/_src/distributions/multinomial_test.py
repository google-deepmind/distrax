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
import jax.numpy as jnp
import numpy as np
from scipy import stats


RTOL = 1e-3


class MultinomialTest(equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(multinomial.Multinomial)
    self.total_count = np.asarray(
        [4, 3], dtype=np.float32)  # float dtype required for TFP
    self.probs = 0.5 * np.asarray([0.1, 0.4, 0.2, 0.3])  # unnormalized
    self.logits = np.log(self.probs)
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  @parameterized.named_parameters(
      ('from probs', False),
      ('from logits', True))
  def test_parameters(self, from_logits):
    if from_logits:
      dist_params = {'logits': self.logits, 'total_count': self.total_count}
    else:
      dist_params = {'probs': self.probs, 'total_count': self.total_count}
    dist = self.distrax_cls(**dist_params)
    self.assertion_fn(dist.logits,
                      np.tile(math.normalize(logits=self.logits), (2, 1)))
    self.assertion_fn(dist.probs,
                      np.tile(math.normalize(probs=self.probs), (2, 1)))

  def test_invalid_parameters(self):
    self._test_raises_error(dist_kwargs={
        'total_count': 3, 'logits': self.logits, 'probs': self.probs})
    self._test_raises_error(
        dist_kwargs={'total_count': 3, 'logits': None, 'probs': None})
    self._test_raises_error(
        dist_kwargs={'total_count': 3, 'logits': 3.}, error_type=AssertionError)
    self._test_raises_error(
        dist_kwargs={'total_count': 3, 'probs': 1.}, error_type=AssertionError)

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
    self.assertion_fn(np.asarray(sum_samples, dtype=np.float32),
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
        assertion_fn=self.assertion_fn)

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

  @parameterized.named_parameters(
      ('bool', jnp.bool_),
      ('complex64', jnp.complex64),
      ('complex128', jnp.complex128))
  def test_invalid_dtype(self, dtype):
    dist_params = {
        'logits': self.logits, 'dtype': dtype, 'total_count': self.total_count}
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

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
        assertion_fn=self.assertion_fn)

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
    entropy = list()
    for probs, counts in zip(dist.probs, dist.total_count):
      entropy.append(stats.multinomial(n=counts, p=probs).entropy())
    self.assertion_fn(
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
        self.variant(dist.entropy)(), expected_result, atol=1e-5)

  @chex.all_variants(with_pmap=False)
  def test_entropy_scalar(self):
    # The TFP Multinomial does not implement `entropy`, so we use scipy for
    # the tests.
    probs = np.asarray([0.1, 0.5, 0.4])
    total_count = 5
    scipy_entropy = stats.multinomial(n=total_count, p=probs).entropy()
    distrax_entropy_fn = self.variant(
        lambda x, y: multinomial.Multinomial._entropy_scalar(total_count, x, y))
    self.assertion_fn(
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
        assertion_fn=self.assertion_fn)

  def test_jittable(self):
    super()._test_jittable(dist_kwargs={
        'probs': np.asarray([1.0, 0.0, 0.0]),
        'total_count': np.asarray([3, 10]),
    })


if __name__ == '__main__':
  absltest.main()
