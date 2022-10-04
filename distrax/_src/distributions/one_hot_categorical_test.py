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
"""Tests for `one_hot_categorical.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import one_hot_categorical
from distrax._src.utils import equivalence
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class OneHotCategoricalTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(one_hot_categorical.OneHotCategorical)
    self.p = np.asarray([0.1, 0.4, 0.2, 0.3])
    self.logits = np.log(self.p) - 1.0  # intended unnormalization

  def test_parameters_from_probs(self):
    dist = self.distrax_cls(probs=self.p)
    self.assertion_fn(rtol=2e-3)(
        dist.logits, math.normalize(logits=np.log(self.p)))
    self.assertion_fn(rtol=2e-3)(dist.probs, math.normalize(probs=self.p))

  def test_parameters_from_logits(self):
    dist = self.distrax_cls(logits=self.logits)
    self.assertion_fn(rtol=2e-3)(
        dist.logits, math.normalize(logits=self.logits))
    self.assertion_fn(rtol=2e-3)(dist.probs, math.normalize(probs=self.p))

  @parameterized.named_parameters(
      ('probs and logits', {'logits': [0.1, -0.2], 'probs': [0.6, 0.4]}),
      ('both probs and logits are None', {'logits': None, 'probs': None}),
      ('bool dtype', {'logits': [0.1, -0.2], 'dtype': jnp.bool_}),
      ('complex64 dtype', {'logits': [0.1, -0.2], 'dtype': jnp.complex64}),
      ('complex128 dtype', {'logits': [0.1, -0.2], 'dtype': jnp.complex128}),
  )
  def test_raises_on_invalid_inputs(self, dist_params):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @chex.all_variants
  def test_negative_probs(self):
    """Check sample returns -1 if probs are negative after normalization."""
    dist = self.distrax_cls(probs=np.asarray([[0.1, -0.4, 0.2, 0.3],
                                              [0.1, 0.1, 0.6, 0.2]]))
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=100))
    samples = sample_fn(self.key)
    self.assertion_fn(rtol=2e-3)(samples[:, 0, :], -1)
    np.testing.assert_array_compare(lambda x, y: x >= y, samples[:, 1, :], 0)

  @chex.all_variants
  def test_nan_probs(self):
    """Checks sample returns -1 if probs are nan after normalization."""
    dist = self.distrax_cls(
        probs=np.asarray([[-0.1, 0.1, 0.0, 0.0], [0.1, 0.1, 0.6, 0.2]]))
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=100))
    samples = sample_fn(self.key)
    self.assertion_fn(rtol=2e-3)(samples[:, 0, :], -1)
    np.testing.assert_array_compare(lambda x, y: x >= y, samples[:, 1, :], 0)

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
      ('1d logits, no shape', {'logits': [0.0, 1.0, -0.5]}, ()),
      ('1d probs, no shape', {'probs': [0.2, 0.5, 0.3]}, ()),
      ('1d logits, int shape', {'logits': [0.0, 1.0, -0.5]}, 1),
      ('1d probs, int shape', {'probs': [0.2, 0.5, 0.3]}, 1),
      ('1d logits, 1-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (1,)),
      ('1d probs, 1-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (1,)),
      ('1d logits, 2-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (5, 4)),
      ('1d probs, 2-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (5, 4)),
      ('2d logits, no shape', {'logits': [[0.0, 1.0, -0.5],
                                          [-0.1, 0.3, 0.0]]}, ()),
      ('2d probs, no shape', {'probs': [[0.1, 0.4, 0.5],
                                        [0.5, 0.25, 0.25]]}, ()),
      ('2d logits, int shape', {'logits': [[0.0, 1.0, -0.5],
                                           [-0.1, 0.3, 0.0]]}, 4),
      ('2d probs, int shape', {'probs': [[0.1, 0.4, 0.5],
                                         [0.5, 0.25, 0.25]]}, 4),
      ('2d logits, 1-tuple shape', {'logits': [[0.0, 1.0, -0.5],
                                               [-0.1, 0.3, 0.0]]}, (5,)),
      ('2d probs, 1-tuple shape', {'probs': [[0.1, 0.4, 0.5],
                                             [0.5, 0.25, 0.25]]}, (5,)),
      ('2d logits, 2-tuple shape', {'logits': [[0.0, 1.0, -0.5],
                                               [-0.1, 0.3, 0.0]]}, (5, 4)),
      ('2d probs, 2-tuple shape', {'probs': [[0.1, 0.4, 0.5],
                                             [0.5, 0.25, 0.25]]}, (5, 4)),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=distr_params,
        sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape', {'logits': [0.0, 1.0, -0.5]}, ()),
      ('1d probs, no shape', {'probs': [0.2, 0.5, 0.3]}, ()),
      ('1d logits, int shape', {'logits': [0.0, 1.0, -0.5]}, 1),
      ('1d probs, int shape', {'probs': [0.2, 0.5, 0.3]}, 1),
      ('1d logits, 1-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (1,)),
      ('1d probs, 1-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (1,)),
      ('1d logits, 2-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (5, 4)),
      ('1d probs, 2-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (5, 4)),
      ('2d logits, no shape', {'logits': [[0.0, 1.0, -0.5],
                                          [-0.1, 0.3, 0.0]]}, ()),
      ('2d probs, no shape', {'probs': [[0.1, 0.4, 0.5],
                                        [0.5, 0.25, 0.25]]}, ()),
      ('2d logits, int shape', {'logits': [[0.0, 1.0, -0.5],
                                           [-0.1, 0.3, 0.0]]}, 4),
      ('2d probs, int shape', {'probs': [[0.1, 0.4, 0.5],
                                         [0.5, 0.25, 0.25]]}, 4),
      ('2d logits, 1-tuple shape', {'logits': [[0.0, 1.0, -0.5],
                                               [-0.1, 0.3, 0.0]]}, (5,)),
      ('2d probs, 1-tuple shape', {'probs': [[0.1, 0.4, 0.5],
                                             [0.5, 0.25, 0.25]]}, (5,)),
      ('2d logits, 2-tuple shape', {'logits': [[0.0, 1.0, -0.5],
                                               [-0.1, 0.3, 0.0]]}, (5, 4)),
      ('2d probs, 2-tuple shape', {'probs': [[0.1, 0.4, 0.5],
                                             [0.5, 0.25, 0.25]]}, (5, 4)),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs=distr_params,
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=2e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('int32', jnp.int32),
      ('int64', jnp.int64),
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist_params = {'logits': self.logits, 'dtype': dtype}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('from probs', False),
      ('from logits', True))
  def test_sample_unique_values(self, from_logits):
    dist_params = {'logits': self.logits} if from_logits else {'probs': self.p}
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=100))
    samples = sample_fn(self.key)
    np.testing.assert_equal(np.unique(samples), np.arange(2))

  @chex.all_variants
  def test_sample_extreme_probs(self):
    dist_params = {'probs': np.asarray([1., 0., 0., 0.])}
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=100))
    samples = sample_fn(self.key)
    np.testing.assert_equal(np.unique(samples[..., 0]), 1)
    np.testing.assert_equal(np.unique(samples[..., 1:]), 0)

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob; 1d logits, 1 input',
       'log_prob',
       {'logits': [0.0, 0.5, -0.5]},
       [1, 0, 0]),
      ('log_prob; 1d logits, 2 inputs',
       'log_prob',
       {'logits': [0.0, 0.5, -0.5]},
       [[1, 0, 0], [0, 1, 0]]),
      ('log_prob; 2d logits, 2 inputs',
       'log_prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.1, 0.1, 0.1]]},
       [[1, 0, 0], [0, 1, 0]]),
      ('log_prob; 2d logits, rank-3 inputs',
       'log_prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.1, 0.1, 0.1]]},
       np.asarray([[1, 0, 0], [0, 1, 0]])[None, ...]),
      ('log_prob; 1d probs, 1 input',
       'log_prob',
       {'probs': [0.3, 0.2, 0.5]},
       [1, 0, 0]),
      ('log_prob; 1d probs, 2 inputs',
       'log_prob',
       {'probs': [0.3, 0.2, 0.5]},
       [[1, 0, 0], [0, 1, 0]]),
      ('log_prob; 2d probs, 2 inputs',
       'log_prob',
       {'probs': [[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]},
       [[1, 0, 0], [0, 1, 0]]),
      ('log_prob; 2d probs, rank-3 inputs',
       'log_prob',
       {'probs': [[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]},
       np.asarray([[1, 0, 0], [0, 1, 0]])[None, ...]),
      ('log_prob; unnormalized probs',
       'log_prob',
       {'probs': [0.1, 0.2, 0.3]},
       [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
      ('prob; 1d logits, 1 input',
       'prob',
       {'logits': [0.0, 0.5, -0.5]},
       [1, 0, 0]),
      ('prob; 1d logits, 2 inputs',
       'prob',
       {'logits': [0.0, 0.5, -0.5]},
       [[1, 0, 0], [0, 1, 0]]),
      ('prob; 2d logits, 2 inputs',
       'prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.1, 0.1, 0.1]]},
       [[1, 0, 0], [0, 1, 0]]),
      ('prob; 2d logits, rank-3 inputs',
       'prob',
       {'logits': [[0.0, 0.5, -0.5], [-0.1, 0.1, 0.1]]},
       np.asarray([[1, 0, 0], [0, 1, 0]])[None, ...]),
      ('prob; 1d probs, 1 input',
       'prob',
       {'probs': [0.3, 0.2, 0.5]},
       [1, 0, 0]),
      ('prob; 1d probs, 2 inputs',
       'prob',
       {'probs': [0.3, 0.2, 0.5]},
       [[1, 0, 0], [0, 1, 0]]),
      ('prob; 2d probs, 2 inputs',
       'prob',
       {'probs': [[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]},
       [[1, 0, 0], [0, 1, 0]]),
      ('prob; 2d probs, rank-3 inputs',
       'prob',
       {'probs': [[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]},
       np.asarray([[1, 0, 0], [0, 1, 0]])[None, ...]),
      ('prob; unnormalized probs',
       'prob',
       {'probs': [0.1, 0.2, 0.3]},
       [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
  )
  def test_pdf(self, function_string, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = np.array(value)
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=2e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob; extreme probs',
       'log_prob',
       {'probs': [0.0, 1.0, 0.0]},
       [[0, 1, 0], [1, 0, 0]],
       np.asarray([0., -np.inf])),
      ('prob; extreme probs',
       'prob',
       {'probs': [0.0, 1.0, 0.0]},
       [[0, 1, 0], [1, 0, 0]],
       np.asarray([1., 0.])),
  )
  def test_pdf_extreme_probs(self, function_string, distr_params,
                             value, expected):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = np.array(value)
    dist = self.distrax_cls(**distr_params)
    self.assertion_fn(rtol=2e-3)(
        self.variant(getattr(dist, function_string))(value), expected)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy; from 2d logits',
       'entropy', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
      ('entropy; from 2d probs',
       'entropy', {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
      ('mode; from 2d logits',
       'mode', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
      ('mode; from 2d probs',
       'mode', {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
  )
  def test_method(self, function_string, distr_params):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=distr_params,
        call_args=(),
        assertion_fn=self.assertion_fn(rtol=2e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('from 2d logits', {
          'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]],
      }, [[0, 1, 0], [1, 0, 0]]),
      ('from 2d probs', {
          'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]],
      }, [[0, 1, 0], [1, 0, 0]]),
  )
  def test_cdf(self, distr_params, values):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    values = np.array(values)
    dist = self.distrax_cls(**distr_params)
    if 'probs' in distr_params:
      probs = distr_params['probs']
    else:
      probs = scipy.special.softmax(distr_params['logits'], axis=-1)
    expected = np.sum(np.cumsum(probs, axis=-1) * values, axis=-1)
    self.assertion_fn(rtol=2e-3)(self.variant(dist.cdf)(values), expected)

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
        dist1_kwargs={'probs': np.array([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]])},
        dist2_kwargs={'logits': np.array([0.0, 0.1, 0.1]),},
        assertion_fn=self.assertion_fn(rtol=2e-3))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_categorical_and_one_hot_categorical(
      self, function_string, mode_string):
    dist1_params = {'probs': np.array([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]])}
    dist2_params = {'logits': np.array([0.0, 0.1, 0.1]),}

    dist1 = categorical.Categorical(**dist1_params)
    tfp_dist1 = tfd.Categorical(**dist1_params)
    dist2 = one_hot_categorical.OneHotCategorical(**dist2_params)
    tfp_dist2 = tfd.OneHotCategorical(**dist2_params)

    distrax_fn_1 = self.variant(getattr(dist1, function_string))
    distrax_fn_2 = self.variant(getattr(dist2, function_string))

    if mode_string == 'distrax_to_distrax':
      comp_dist1_dist2 = distrax_fn_1(dist2)
      comp_dist2_dist1 = distrax_fn_2(dist1)
    elif mode_string == 'distrax_to_tfp':
      comp_dist1_dist2 = distrax_fn_1(tfp_dist2)
      comp_dist2_dist1 = distrax_fn_2(tfp_dist1)
    elif mode_string == 'tfp_to_distrax':
      comp_dist1_dist2 = getattr(tfp_dist1, function_string)(dist2)
      comp_dist2_dist1 = getattr(tfp_dist2, function_string)(dist1)

    # The target values (obtained with TFP-only methods) are obtained with two
    # distributions of the same class (namely, Categorical) because TFP doesn't
    # register KLs of the form KL(Categorical || OneHotCategorical).
    tfp_dist2_aux = tfd.Categorical(**dist2_params)
    tfp_comp_dist1_dist2 = getattr(tfp_dist1, function_string)(tfp_dist2_aux)
    tfp_comp_dist2_dist1 = getattr(tfp_dist2_aux, function_string)(tfp_dist1)

    self.assertion_fn(rtol=2e-3)(comp_dist1_dist2, tfp_comp_dist1_dist2)
    self.assertion_fn(rtol=2e-3)(comp_dist2_dist1, tfp_comp_dist2_dist1)

  def test_jittable(self):
    super()._test_jittable((np.zeros((3,)),))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
  )
  def test_slice(self, slice_):
    logits = jnp.array(np.random.randn(3, 4, 5))
    probs = jax.nn.softmax(jnp.array(np.random.randn(3, 4, 5)), axis=-1)
    dist1 = self.distrax_cls(logits=logits)
    dist2 = self.distrax_cls(probs=probs)
    dist1_sliced = dist1[slice_]
    dist2_sliced = dist2[slice_]
    self.assertion_fn(rtol=2e-3)(
        jax.nn.softmax(dist1_sliced.logits, axis=-1),
        jax.nn.softmax(logits[slice_], axis=-1))
    self.assertion_fn(rtol=2e-3)(dist2_sliced.probs, probs[slice_])
    self.assertIsInstance(dist1_sliced, one_hot_categorical.OneHotCategorical)
    self.assertIsInstance(dist2_sliced, one_hot_categorical.OneHotCategorical)

  def test_slice_ellipsis(self):
    logits = jnp.array(np.random.randn(4, 4, 5))
    probs = jax.nn.softmax(jnp.array(np.random.randn(4, 4, 5)), axis=-1)
    dist1 = self.distrax_cls(logits=logits)
    dist2 = self.distrax_cls(probs=probs)
    self.assertion_fn(rtol=2e-3)(
        jax.nn.softmax(dist1[..., -1].logits, axis=-1),
        jax.nn.softmax(logits[:, -1], axis=-1))
    self.assertion_fn(rtol=2e-3)(dist2[..., -1].probs, probs[:, -1])


if __name__ == '__main__':
  absltest.main()
