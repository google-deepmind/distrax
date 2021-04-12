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
"""Tests for `categorical.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import categorical
from distrax._src.utils import equivalence
from distrax._src.utils import math
import jax.numpy as jnp
import numpy as np


RTOL = 2e-3


class CategoricalTest(equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(categorical.Categorical)
    self.p = np.asarray([0.1, 0.4, 0.2, 0.3])
    self.logits = np.log(self.p) - 1.0  # intended unnormalization
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  def test_parameters_from_probs(self):
    dist = self.distrax_cls(probs=self.p)
    self.assertion_fn(dist.logits, math.normalize(logits=np.log(self.p)))
    self.assertion_fn(dist.probs, math.normalize(probs=self.p))

  def test_parameters_from_logits(self):
    dist = self.distrax_cls(logits=self.logits)
    self.assertion_fn(dist.logits, math.normalize(logits=self.logits))
    self.assertion_fn(dist.probs, math.normalize(probs=self.p))

  def test_invalid_parameters(self):
    self._test_raises_error(
        dist_kwargs={'logits': self.logits, 'probs': self.p})
    self._test_raises_error(dist_kwargs={'logits': None, 'probs': None})

  @parameterized.named_parameters(
      ('from probs', False),
      ('from logits', True))
  def test_num_categories(self, from_logits):
    dist_params = {'logits': self.logits} if from_logits else {'probs': self.p}
    dist = self.distrax_cls(**dist_params)
    np.testing.assert_equal(dist.num_categories, len(self.p))

  @parameterized.named_parameters(
      ('1d logits', {'logits': [0.0, 1.0, -0.5]}),
      ('1d probs', {'probs': [0.2, 0.5, 0.3]}),
      ('2d logits', {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]}),
      ('2d probs', {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]}),
  )
  def test_event_shape(self, distr_params):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_event_shape((), distr_params)

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
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=distr_params,
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
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs=distr_params,
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
    dist_params = {'logits': self.logits, 'dtype': dtype}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @parameterized.named_parameters(
      ('bool', jnp.bool_),
      ('complex64', jnp.complex64),
      ('complex128', jnp.complex128))
  def test_invalid_dtype(self, dtype):
    dist_params = {'logits': self.logits, 'dtype': dtype}
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      ('from probs', False),
      ('from logits', True))
  def test_sample_unique_values(self, from_logits):
    dist_params = {'logits': self.logits} if from_logits else {'probs': self.p}
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=1000))
    samples = sample_fn(self.key)
    np.testing.assert_equal(np.unique(samples), np.arange(len(self.p)))

  @chex.all_variants
  def test_sample_extreme_probs(self):
    dist_params = {'probs': np.asarray([0., 0., 1., 0.])}
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=100))
    samples = sample_fn(self.key)
    np.testing.assert_equal(np.unique(samples),
                            np.argmax(dist_params['probs']).astype(np.int32))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob; 1d logits, 1d value',
       'log_prob',
       {'logits': [0.0, 0.5, -0.5]},
       [1, 0, 2, 0]),
      ('log_prob; 1d probs, 1d value',
       'log_prob',
       {'probs': [0.3, 0.2, 0.5]},
       [1, 0, 2, 0]),
      ('log_prob; 1d logits, 2d value',
       'log_prob',
       {'logits': [0.0, 0.5, -0.5]},
       [[1, 0], [2, 0]]),
      ('log_prob; 1d probs, 2d value',
       'log_prob',
       {'probs': [0.3, 0.2, 0.5]},
       [[1, 0], [2, 0]]),
      ('log_prob; 2d logits, 1d value',
       'log_prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [1, 0]),
      ('log_prob; 2d probs, 1d value',
       'log_prob',
       {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [1, 0]),
      ('log_prob; 2d logits, 2d value',
       'log_prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [[1, 0], [2, 1]]),
      ('log_prob; 2d probs, 2d value',
       'log_prob',
       {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [[1, 0], [2, 1]]),
      ('log_prob; extreme probs',
       'log_prob',
       {'probs': [0.0, 1.0, 0.0]},
       [0, 1, 1, 2]),
      ('log_prob; unnormalized probs',
       'log_prob',
       {'probs': [0.1, 0.2, 0.3]},
       [0, 1, 2]),
      ('prob; 1d logits, 1d value',
       'prob',
       {'logits': [0.0, 0.5, -0.5]},
       [1, 0, 2, 0]),
      ('prob; 1d probs, 1d value',
       'prob',
       {'probs': [0.3, 0.2, 0.5]},
       [1, 0, 2, 0]),
      ('prob; 1d logits, 2d value',
       'prob',
       {'logits': [0.0, 0.5, -0.5]},
       [[1, 0], [2, 0]]),
      ('prob; 1d probs, 2d value',
       'prob',
       {'probs': [0.3, 0.2, 0.5]},
       [[1, 0], [2, 0]]),
      ('prob; 2d logits, 1d value',
       'prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [1, 0]),
      ('prob; 2d probs, 1d value',
       'prob',
       {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [1, 0]),
      ('prob; 2d logits, 2d value',
       'prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [[1, 0], [2, 1]]),
      ('prob; 2d probs, 2d value',
       'prob',
       {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [[1, 0], [2, 1]]),
      ('prob; extreme probs',
       'prob',
       {'probs': [0.0, 1.0, 0.0]},
       [0, 1, 1, 2]),
      ('prob; unnormalized probs',
       'prob',
       {'probs': [0.1, 0.2, 0.3]},
       [0, 1, 2]),
  )
  def test_pdf(self, function_string, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = np.asarray(value, dtype=np.int32)
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy; from 2d logits',
       'entropy',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       ()),
      ('entropy; from 2d probs',
       'entropy',
       {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]},
       ()),
      ('mode; from 2d logits',
       'mode',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       ()),
      ('mode; from 2d probs',
       'mode',
       {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]},
       ()),
      ('cdf; from 2d logits',
       'cdf',
       {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       ([[0, 1], [2, 1]],)),
      ('cdf; from 2d probs',
       'cdf',
       {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]},
       ([[0, 1], [2, 1]],)),
  )
  def test_method(self, function_string, distr_params, values):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    values = () if not values else jnp.asarray(values, dtype=jnp.int32)
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=distr_params,
        call_args=values,
        assertion_fn=self.assertion_fn)

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
        dist1_kwargs={'probs': jnp.asarray([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]])},
        dist2_kwargs={'logits': jnp.asarray([0.0, 0.1, 0.1]),},
        assertion_fn=self.assertion_fn)

  def test_jittable(self):
    super()._test_jittable((np.array([0., 4., -1., 4.]),))


if __name__ == '__main__':
  absltest.main()
