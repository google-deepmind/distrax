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
import jax
import jax.numpy as jnp
import numpy as np


class CategoricalTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(categorical.Categorical)

  @parameterized.named_parameters(
      ('1d probs', (4,), True),
      ('1d logits', (4,), False),
      ('2d probs', (3, 4), True),
      ('2d logits', (3, 4), False),
  )
  def test_properties(self, shape, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=shape)  # Intentional unnormalization of `probs`.
    logits = np.log(probs)
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, shape[:-1])
    self.assertEqual(dist.num_categories, shape[-1])
    self.assertion_fn(rtol=1e-3)(dist.logits, math.normalize(logits=logits))
    self.assertion_fn(rtol=1e-3)(dist.probs, math.normalize(probs=probs))

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
  @parameterized.named_parameters(
      ('1d logits, no shape', {'logits': [0.0, 1.0, -0.5]}, ()),
      ('1d probs, no shape', {'probs': [0.2, 0.5, 0.3]}, ()),
      ('1d logits, int shape', {'logits': [0.0, 1.0, -0.5]}, 1),
      ('1d probs, int shape', {'probs': [0.2, 0.5, 0.3]}, 1),
      ('1d logits, 1-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (1,)),
      ('1d probs, 1-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (1,)),
      ('1d logits, 2-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (5, 4)),
      ('1d probs, 2-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (5, 4)),
      ('2d logits, no shape', {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape', {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape', {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape', {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]}, (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]}, (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]}, (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]}, (5, 4)),
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
      ('2d logits, no shape', {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       ()),
      ('2d probs, no shape', {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       ()),
      ('2d logits, int shape', {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]},
       4),
      ('2d probs, int shape', {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
       4),
      ('2d logits, 1-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]}, (5,)),
      ('2d probs, 1-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]}, (5,)),
      ('2d logits, 2-tuple shape',
       {'logits': [[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]]}, (5, 4)),
      ('2d probs, 2-tuple shape',
       {'probs': [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]}, (5, 4)),
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
      ('sample, int16', 'sample', jnp.int16),
      ('sample, int32', 'sample', jnp.int32),
      ('sample, uint16', 'sample', jnp.uint16),
      ('sample, uint32', 'sample', jnp.uint32),
      ('sample, float16', 'sample', jnp.float16),
      ('sample, float32', 'sample', jnp.float32),
      ('sample_and_log_prob, int16', 'sample_and_log_prob', jnp.int16),
      ('sample_and_log_prob, int32', 'sample_and_log_prob', jnp.int32),
      ('sample_and_log_prob, uint16', 'sample_and_log_prob', jnp.uint16),
      ('sample_and_log_prob, uint32', 'sample_and_log_prob', jnp.uint32),
      ('sample_and_log_prob, float16', 'sample_and_log_prob', jnp.float16),
      ('sample_and_log_prob, float32', 'sample_and_log_prob', jnp.float32),
  )
  def test_sample_dtype(self, method, dtype):
    dist_params = {'logits': [0.1, -0.1, 0.5, -0.8, 1.5], 'dtype': dtype}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(getattr(dist, method))(seed=self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.dtype, dist.dtype)
    self.assertEqual(samples.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, from probs', 'sample', True),
      ('sample, from logits', 'sample', False),
      ('sample_and_log_prob, from probs', 'sample_and_log_prob', True),
      ('sample_and_log_prob, from logits', 'sample_and_log_prob', False),
  )
  def test_sample_values(self, method, from_probs):
    probs = np.array([[0.5, 0.25, 0.25], [0., 0., 1.]])  # Includes edge case.
    num_categories = probs.shape[-1]
    logits = np.log(probs)
    n_samples = 100000
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    sample_fn = self.variant(
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples))
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.shape, (n_samples,) + probs.shape[:-1])
    self.assertTrue(np.all(
        np.logical_and(samples >= 0, samples < num_categories)))
    np.testing.assert_array_equal(jnp.floor(samples), samples)
    samples_one_hot = jax.nn.one_hot(samples, num_categories, axis=-1)
    self.assertion_fn(rtol=0.1)(np.mean(samples_one_hot, axis=0), probs)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample', 'sample'),
      ('sample_and_log_prob', 'sample_and_log_prob'),
  )
  def test_sample_values_invalid_probs(self, method):
    # Check that samples=-1 if probs are negative or NaN after normalization.
    n_samples = 1000
    probs = np.array([
        [0.1, -0.4, 0.2, 0.3],  # Negative probabilities.
        [-0.1, 0.1, 0.0, 0.0],  # NaN probabilities after normalization.
        [0.1, 0.25, 0.2, 0.8],  # Valid (unnormalized) probabilities.
    ])
    dist = self.distrax_cls(probs=probs)
    sample_fn = self.variant(
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples))
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertion_fn(rtol=1e-4)(samples[..., :-1], -1)
    np.testing.assert_array_compare(lambda x, y: x >= y, samples[..., -1], 0)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, 1d value', {'logits': [0.0, 0.5, -0.5]}, [1, 0, 2, 0]),
      ('1d probs, 1d value', {'probs': [0.3, 0.2, 0.5]}, [1, 0, 2, 0]),
      ('1d logits, 2d value', {'logits': [0.0, 0.5, -0.5]}, [[1, 0], [2, 0]]),
      ('1d probs, 2d value', {'probs': [0.3, 0.2, 0.5]}, [[1, 0], [2, 0]]),
      ('2d logits, 1d value', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [1, 0]),
      ('2d probs, 1d value', {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [1, 0]),
      ('2d logits, 2d value', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [[1, 0], [2, 1]]),
      ('2d probs, 2d value', {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [[1, 0], [2, 1]]),
      ('extreme probs', {'probs': [0.0, 1.0, 0.0]}, [0, 1, 1, 2]),
  )
  def test_method_with_input(self, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = np.asarray(value, dtype=np.int32)
    for method in ['prob', 'log_prob', 'cdf', 'log_cdf', 'survival_function']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(atol=3e-5))
    # We separate the `log_survival_function` method because TFP sometimes gives
    # NaN instead of `-jnp.inf` when evaluated at `num_categories - 1` on a TPU
    # if the distribution was created using the logits parameter.
    dist = self.distrax_cls(**distr_params)
    tfp_dist = self.distrax_cls.equiv_tfp_cls(**distr_params)
    num_categories = dist.num_categories
    log_sf = tfp_dist.log_survival_function(value)
    log_sf = jnp.where(value == num_categories - 1, -jnp.inf, log_sf)
    with self.subTest(method='log_survival_function'):
      self.assertion_fn(atol=3e-5)(
          self.variant(dist.log_survival_function)(value), log_sf)

  @chex.all_variants
  def test_method_with_input_unnormalized_probs(self):
    # We test this case separately because the result of `cdf` and `log_cdf`
    # differs from TFP when the input `probs` are not normalized.
    probs = np.array([0.1, 0.2, 0.3])
    normalized_probs = probs / np.sum(probs, axis=-1, keepdims=True)
    distr_params = {'probs': probs}
    value = np.asarray([0, 1, 2], dtype=np.int32)
    dist = self.distrax_cls(**distr_params)
    self.assertion_fn(rtol=1e-3)(
        self.variant(dist.prob)(value), normalized_probs)
    self.assertion_fn(rtol=1e-3)(
        self.variant(dist.log_prob)(value), np.log(normalized_probs))
    self.assertion_fn(rtol=1e-3)(
        self.variant(dist.cdf)(value), np.cumsum(normalized_probs))
    self.assertion_fn(atol=5e-5)(
        self.variant(dist.log_cdf)(value), np.log(np.cumsum(normalized_probs)))
    self.assertion_fn(atol=1e-5)(
        self.variant(dist.survival_function)(value),
        1. - np.cumsum(normalized_probs))
    # In the line below, we compare against `jnp` instead of `np` because the
    # latter gives `1. - np.cumsum(normalized_probs)[-1] = 1.1e-16` instead of
    # `0.`, so its log is innacurate: it gives `-36.7` instead of `-np.inf`.
    self.assertion_fn(atol=1e-5)(
        self.variant(dist.log_survival_function)(value),
        jnp.log(1. - jnp.cumsum(normalized_probs)))

  @chex.all_variants
  def test_method_with_input_outside_domain(self):
    probs = jnp.asarray([0.2, 0.3, 0.5])
    dist = self.distrax_cls(probs=probs)
    value = jnp.asarray([-1, -2, 3, 4], dtype=jnp.int32)
    self.assertion_fn(atol=1e-5)(
        self.variant(dist.prob)(value), np.asarray([0., 0., 0., 0.]))
    self.assertTrue(np.all(self.variant(dist.log_prob)(value) == -jnp.inf))
    self.assertion_fn(atol=1e-5)(
        self.variant(dist.cdf)(value), np.asarray([0., 0., 1., 1.]))
    self.assertion_fn(rtol=1e-3)(
        self.variant(dist.log_cdf)(value), np.log(np.asarray([0., 0., 1., 1.])))
    self.assertion_fn(atol=1e-5)(
        self.variant(dist.survival_function)(value),
        np.asarray([1., 1., 0., 0.]))
    self.assertion_fn(atol=1e-5)(
        self.variant(dist.log_survival_function)(value),
        np.log(np.asarray([1., 1., 0., 0.])))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('2d probs', True),
      ('2d logits', False),
  )
  def test_method(self, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=(4, 3))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    logits = np.log(probs)
    distr_params = {'probs': probs} if from_probs else {'logits': logits}
    for method in ['entropy', 'mode', 'logits_parameter']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(),
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
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'probs':
                jnp.asarray([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]])
        },
        dist2_kwargs={
            'logits': jnp.asarray([0.0, 0.1, 0.1]),
        },
        assertion_fn=self.assertion_fn(rtol=2e-3))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'),
  )
  def test_with_two_distributions_extreme_cases(
      self, function_string, mode_string):
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'probs':
                jnp.asarray([[0.1, 0.5, 0.4], [0.4, 0.0, 0.6], [0.4, 0.6, 0.]])
        },
        dist2_kwargs={
            'logits': jnp.asarray([0.0, 0.1, -jnp.inf]),
        },
        assertion_fn=self.assertion_fn(atol=5e-5))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'),
  )
  def test_with_two_distributions_raises_on_invalid_num_categories(
      self, function_string, mode_string):
    probs1 = jnp.asarray([0.1, 0.5, 0.4])
    distrax_dist1 = self.distrax_cls(probs=probs1)
    tfp_dist1 = self.distrax_cls.equiv_tfp_cls(probs=probs1)
    logits2 = jnp.asarray([-0.1, 0.3])
    distrax_dist2 = self.distrax_cls(logits=logits2)
    tfp_dist2 = self.distrax_cls.equiv_tfp_cls(logits=logits2)
    dist_a = tfp_dist1 if mode_string == 'tfp_to_distrax' else distrax_dist1
    dist_b = tfp_dist2 if mode_string == 'distrax_to_tfp' else distrax_dist2
    first_fn = self.variant(getattr(dist_a, function_string))
    with self.assertRaises(ValueError):
      _ = first_fn(dist_b)
    dist_a = tfp_dist2 if mode_string == 'tfp_to_distrax' else distrax_dist2
    dist_b = tfp_dist1 if mode_string == 'distrax_to_tfp' else distrax_dist1
    second_fn = self.variant(getattr(dist_a, function_string))
    with self.assertRaises(ValueError):
      _ = second_fn(dist_b)

  def test_jittable(self):
    super()._test_jittable((np.array([0., 4., -1., 4.]),))

  @parameterized.named_parameters(
      ('single element, from probs', 2, True),
      ('single element, from logits', 2, False),
      ('range, from probs', slice(-1), True),
      ('range, from logits', slice(-1), False),
      ('range_2, from probs', (slice(None), slice(-1)), True),
      ('range_2, from logits', (slice(None), slice(-1)), False),
  )
  def test_slice(self, slice_, from_probs):
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(3, 4, 5))
    probs = jax.nn.softmax(logits, axis=-1)
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(
        dist[slice_].logits, math.normalize(logits=logits[slice_]))
    self.assertion_fn(rtol=1e-3)(dist[slice_].probs, probs[slice_])

  @parameterized.named_parameters(
      ('from probs', True),
      ('from logits', False),
  )
  def test_slice_ellipsis(self, from_probs):
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(3, 4, 5))
    probs = jax.nn.softmax(logits, axis=-1)
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(
        dist[..., -1].logits, math.normalize(logits=logits[:, -1, :]))
    self.assertion_fn(rtol=1e-3)(dist[..., -1].probs, probs[:, -1, :])

  def test_vmap_inputs(self):
    def log_prob_sum(dist, x):
      return dist.log_prob(x).sum()

    dist = categorical.Categorical(jnp.arange(3 * 4 * 5).reshape((3, 4, 5)))
    x = jnp.zeros((3, 4), jnp.int_)

    with self.subTest('no vmap'):
      actual = log_prob_sum(dist, x)
      expected = dist.log_prob(x).sum()
      self.assertion_fn()(actual, expected)

    with self.subTest('axis=0'):
      actual = jax.vmap(log_prob_sum, in_axes=0)(dist, x)
      expected = dist.log_prob(x).sum(axis=1)
      self.assertion_fn()(actual, expected)

    with self.subTest('axis=1'):
      actual = jax.vmap(log_prob_sum, in_axes=1)(dist, x)
      expected = dist.log_prob(x).sum(axis=0)
      self.assertion_fn()(actual, expected)

  def test_vmap_outputs(self):
    def summed_dist(logits):
      logits1 = logits.sum(keepdims=True)
      logits2 = -logits1
      logits = jnp.concatenate([logits1, logits2], axis=-1)
      return categorical.Categorical(logits)

    logits = jnp.arange((3 * 4 * 5)).reshape((3, 4, 5))
    actual = jax.vmap(summed_dist)(logits)

    logits1 = logits.sum(axis=(1, 2), keepdims=True)
    logits2 = -logits1
    logits = jnp.concatenate([logits1, logits2], axis=-1)
    expected = categorical.Categorical(logits)

    np.testing.assert_equal(actual.batch_shape, expected.batch_shape)
    np.testing.assert_equal(actual.event_shape, expected.event_shape)

    x = jnp.array([[[0]], [[1]], [[0]]], jnp.int_)
    self.assertion_fn(rtol=1e-6)(actual.log_prob(x), expected.log_prob(x))

  @parameterized.named_parameters(
      ('-inf logits', np.array([-jnp.inf, 2, -3, -jnp.inf, 5.0])),
      ('uniform large negative logits', np.array([-1e9] * 11)),
      ('uniform large positive logits', np.array([1e9] * 11)),
      ('uniform', np.array([0.0] * 11)),
      ('typical', np.array([1, 7, -3, 2, 4.0])),
  )
  def test_entropy_grad(self, logits):
    clipped_logits = jnp.maximum(-10000, logits)

    def entropy_fn(logits):
      return categorical.Categorical(logits).entropy()
    entropy, grads = jax.value_and_grad(entropy_fn)(logits)
    expected_entropy, expected_grads = jax.value_and_grad(entropy_fn)(
        clipped_logits)
    self.assertion_fn(rtol=1e-6)(expected_entropy, entropy)
    self.assertion_fn(rtol=1e-6)(expected_grads, grads)
    self.assertTrue(np.isfinite(entropy).all())
    self.assertTrue(np.isfinite(grads).all())

  @parameterized.named_parameters(
      ('-inf logits1', np.array([-jnp.inf, 2, -3, -jnp.inf, 5.0]),
       np.array([1, 7, -3, 2, 4.0])),
      ('-inf logits both', np.array([-jnp.inf, 2, -1000, -jnp.inf, 5.0]),
       np.array([-jnp.inf, 7, -jnp.inf, 2, 4.0])),
      ('typical', np.array([5, -2, 0, 1, 4.0]),
       np.array([1, 7, -3, 2, 4.0])),
  )
  def test_kl_grad(self, logits1, logits2):
    clipped_logits1 = jnp.maximum(-10000, logits1)
    clipped_logits2 = jnp.maximum(-10000, logits2)

    def kl_fn(logits1, logits2):
      return categorical.Categorical(logits1).kl_divergence(
          categorical.Categorical(logits2))
    kl, grads = jax.value_and_grad(
        kl_fn, argnums=(0, 1))(logits1, logits2)
    expected_kl, expected_grads = jax.value_and_grad(
        kl_fn, argnums=(0, 1))(clipped_logits1, clipped_logits2)
    self.assertion_fn(rtol=1e-6)(expected_kl, kl)
    self.assertion_fn(rtol=1e-6)(expected_grads, grads)
    self.assertTrue(np.isfinite(kl).all())
    self.assertTrue(np.isfinite(grads).all())


if __name__ == '__main__':
  absltest.main()
