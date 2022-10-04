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
"""Tests for `bernoulli.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import bernoulli
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np
from scipy import special as sp_special


class BernoulliTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(bernoulli.Bernoulli)
    self.p = np.asarray([0.2, 0.4, 0.6, 0.8])
    self.logits = sp_special.logit(self.p)

  @parameterized.named_parameters(
      ('0d probs', (), True),
      ('0d logits', (), False),
      ('1d probs', (4,), True),
      ('1d logits', (4,), False),
      ('2d probs', (3, 4), True),
      ('2d logits', (3, 4), False),
  )
  def test_properties(self, shape, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=shape)
    logits = sp_special.logit(probs)
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.logits, logits)
    self.assertion_fn(rtol=1e-3)(dist.probs, probs)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, shape)

  @parameterized.named_parameters(
      ('probs and logits', {'logits': [0.1, -0.2], 'probs': [0.5, 0.4]}),
      ('both probs and logits are None', {'logits': None, 'probs': None}),
      ('complex64 dtype', {'logits': [0.1, -0.2], 'dtype': jnp.complex64}),
      ('complex128 dtype', {'logits': [0.1, -0.2], 'dtype': jnp.complex128}),
  )
  def test_raises_on_invalid_inputs(self, dist_params):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape', {'logits': [0.0, 1.0, -0.5]}, ()),
      ('1d probs, no shape', {'probs': [0.1, 0.5, 0.3]}, ()),
      ('1d logits, int shape', {'logits': [0.0, 1.0, -0.5]}, 1),
      ('1d probs, int shape', {'probs': [0.1, 0.5, 0.3]}, 1),
      ('1d logits, 1-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (1,)),
      ('1d probs, 1-tuple shape', {'probs': [0.1, 0.5, 0.3]}, (1,)),
      ('1d logits, 2-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (5, 4)),
      ('1d probs, 2-tuple shape', {'probs': [0.1, 0.5, 0.3]}, (5, 4)),
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
      ('sample, from probs', 'sample', True),
      ('sample, from logits', 'sample', False),
      ('sample_and_log_prob, from probs', 'sample_and_log_prob', True),
      ('sample_and_log_prob, from logits', 'sample_and_log_prob', False),
  )
  def test_sample_values(self, method, from_probs):
    probs = np.array([0., 0.2, 0.5, 0.8, 1.])  # Includes edge cases (0 and 1).
    logits = sp_special.logit(probs)
    n_samples = 100000
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    sample_fn = self.variant(
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples))
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.shape, (n_samples,) + probs.shape)
    self.assertTrue(np.all(np.logical_or(samples == 0, samples == 1)))
    self.assertion_fn(rtol=0.1)(np.mean(samples, axis=0), probs)
    self.assertion_fn(atol=2e-3)(np.std(samples, axis=0), dist.stddev())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape', {'logits': [0.0, 1.0, -0.5]}, ()),
      ('1d probs, no shape', {'probs': [0.1, 0.5, 0.3]}, ()),
      ('1d logits, int shape', {'logits': [0.0, 1.0, -0.5]}, 1),
      ('1d probs, int shape', {'probs': [0.1, 0.5, 0.3]}, 1),
      ('1d logits, 1-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (1,)),
      ('1d probs, 1-tuple shape', {'probs': [0.1, 0.5, 0.3]}, (1,)),
      ('1d logits, 2-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (5, 4)),
      ('1d probs, 2-tuple shape', {'probs': [0.1, 0.5, 0.3]}, (5, 4)),
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
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, bool', 'sample', jnp.bool_),
      ('sample, uint16', 'sample', jnp.uint16),
      ('sample, uint32', 'sample', jnp.uint32),
      ('sample, int16', 'sample', jnp.int16),
      ('sample, int32', 'sample', jnp.int32),
      ('sample, float16', 'sample', jnp.float16),
      ('sample, float32', 'sample', jnp.float32),
      ('sample_and_log_prob, bool', 'sample_and_log_prob', jnp.bool_),
      ('sample_and_log_prob, uint16', 'sample_and_log_prob', jnp.uint16),
      ('sample_and_log_prob, uint32', 'sample_and_log_prob', jnp.uint32),
      ('sample_and_log_prob, int16', 'sample_and_log_prob', jnp.int16),
      ('sample_and_log_prob, int32', 'sample_and_log_prob', jnp.int32),
      ('sample_and_log_prob, float16', 'sample_and_log_prob', jnp.float16),
      ('sample_and_log_prob, float32', 'sample_and_log_prob', jnp.float32),
  )
  def test_sample_dtype(self, method, dtype):
    dist_params = {'logits': self.logits, 'dtype': dtype}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(getattr(dist, method))(seed=self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.dtype, dist.dtype)
    self.assertEqual(samples.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, int value', {'logits': [0.0, 0.5, -0.5]}, 1),
      ('1d probs, int value', {'probs': [0.3, 0.2, 0.5]}, 1),
      ('1d logits, 1d value', {'logits': [0.0, 0.5, -0.5]}, [1, 0, 1]),
      ('1d probs, 1d value', {'probs': [0.3, 0.2, 0.5]}, [1, 0, 1]),
      ('1d logits, 2d value', {'logits': [0.0, 0.5, -0.5]},
       [[1, 0, 0], [0, 1, 0]]),
      ('1d probs, 2d value', {'probs': [0.3, 0.2, 0.5]},
       [[1, 0, 0], [0, 1, 0]]),
      ('2d logits, 1d value', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [1, 0, 1]),
      ('2d probs, 1d value', {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [1, 0, 1]),
      ('2d logits, 2d value', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
       [[1, 0, 0], [1, 1, 0]]),
      ('2d probs, 2d value', {'probs': [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
       [[1, 0, 0], [1, 1, 0]]),
      ('edge cases with logits', {'logits': [-np.inf, -np.inf, np.inf, np.inf]},
       [0, 1, 0, 1]),
      ('edge cases with probs', {'probs': [0.0, 0.0, 1.0, 1.0]}, [0, 1, 0, 1]),
  )
  def test_method_with_value(self, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = jnp.asarray(value)
    for method in ['prob', 'log_prob', 'cdf', 'log_cdf',
                   'survival_function', 'log_survival_function']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('from logits', {'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
      ('from probs', {'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
  )
  def test_method(self, distr_params):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    for method in ['entropy', 'mode', 'mean', 'variance', 'stddev']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(),
            assertion_fn=self.assertion_fn(rtol=1e-2))

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
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'probs': jnp.asarray([[0.1, 0.5, 0.4], [0.2, 0.4, 0.8]])},
        dist2_kwargs={'logits': jnp.asarray([0.0, -0.1, 0.1]),},
        assertion_fn=self.assertion_fn(rtol=1e-2))

  def test_jittable(self):
    super()._test_jittable(
        (np.array([0., 4., -1., 4.]),),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @parameterized.named_parameters(
      ('single element, from probs', 2, True),
      ('single element, from logits', 2, False),
      ('range, from probs', slice(-1), True),
      ('range, from logits', slice(-1), False),
      ('range_2, from probs', (slice(None), slice(-1)), True),
      ('range_2, from logits', (slice(None), slice(-1)), False),
      ('ellipsis, from probs', (Ellipsis, -1), True),
      ('ellipsis, from logits', (Ellipsis, -1), False),
  )
  def test_slice(self, slice_, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=(3, 4, 5))
    logits = sp_special.logit(probs)
    dist_kwargs = {'probs': probs} if from_probs else {'logits': logits}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist[slice_].logits, logits[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].probs, probs[slice_])


if __name__ == '__main__':
  absltest.main()
