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
from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions import binomial
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np
from scipy import special as sp_special


class BinomialTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(binomial.Binomial)
    self.n = np.array([5, 10, 20])
    self.p = np.asarray([0.1, 0.4, 0.9])
    self.logits = sp_special.logit(self.p)

  @parameterized.named_parameters(
      ('0d params, from probs', (), True),
      ('0d params, from logits', (), False),
      ('1d params, from probs', (3,), True),
      ('1d params, from logits', (3,), False),
      ('2d params, from probs', (3, 4), True),
      ('2d params, from logits', (3, 4), False),
  )
  def test_properties(self, shape, from_probs):
    rng = np.random.default_rng(42)
    total_count = rng.integers(low=1, high=100, size=shape)
    probs = rng.uniform(size=shape)
    logits = sp_special.logit(probs)

    dist_kwargs = {'total_count': total_count}
    if from_probs:
      dist_kwargs['probs'] = probs
    else:
      dist_kwargs['logits'] = logits

    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.logits, logits)
    self.assertion_fn(rtol=1e-3)(dist.probs, probs)
    self.assertion_fn(rtol=1e-3)(dist.total_count, total_count)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, shape)

  @parameterized.named_parameters(
      ('probs and logits', {'total_count': 5, 'logits': [0.1], 'probs': [0.5]}),
      (
          'both probs and logits are None',
          {'total_count': 5, 'logits': None, 'probs': None},
      ),
      (
          'complex64 dtype',
          {'total_count': 5, 'logits': [0.1], 'dtype': jnp.complex64},
      ),
      ('non-integer total_count', {'total_count': 5.5, 'probs': [0.1]}),
  )
  def test_raises_on_invalid_inputs(self, dist_params):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      (
          '1d logits, no shape',
          {'total_count': 10, 'logits': [0.0, 1.0, -0.5]},
          (),
      ),
      ('1d probs, no shape', {'total_count': 10, 'probs': [0.1, 0.5, 0.3]}, ()),
      (
          '1d logits, int shape',
          {'total_count': 10, 'logits': [0.0, 1.0, -0.5]},
          1,
      ),
      ('1d probs, int shape', {'total_count': 10, 'probs': [0.1, 0.5, 0.3]}, 1),
      (
          '1d logits, 1-tuple shape',
          {'total_count': 10, 'logits': [0.0, 1.0, -0.5]},
          (1,),
      ),
      (
          '1d probs, 1-tuple shape',
          {'total_count': 10, 'probs': [0.1, 0.5, 0.3]},
          (1,),
      ),
      (
          '1d logits, 2-tuple shape',
          {'total_count': 10, 'logits': [0.0, 1.0, -0.5]},
          (5, 4),
      ),
      (
          '1d probs, 2-tuple shape',
          {'total_count': 10, 'probs': [0.1, 0.5, 0.3]},
          (5, 4),
      ),
      (
          'broadcast n, 1d probs',
          {'total_count': [5, 10], 'probs': [0.1, 0.5]},
          (5, 4),
      ),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_sample_shape(
        dist_args=(), dist_kwargs=distr_params, sample_shape=sample_shape
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, from probs', 'sample', True),
      ('sample, from logits', 'sample', False),
      ('sample_and_log_prob, from probs', 'sample_and_log_prob', True),
      ('sample_and_log_prob, from logits', 'sample_and_log_prob', False),
  )
  def test_sample_values(self, method, from_probs):
    total_count = np.array([5, 10])
    probs = np.array([0.1, 0.9])  # Includes edge case (0.9 > 0.5)
    logits = sp_special.logit(probs)
    n_samples = 100000

    dist_kwargs = {'total_count': total_count}
    if from_probs:
      dist_kwargs['probs'] = probs
    else:
      dist_kwargs['logits'] = logits

    dist = self.distrax_cls(**dist_kwargs)
    sample_fn = self.variant(  # pytype: disable=attribute-error
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples)
    )
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples

    self.assertEqual(samples.shape, (n_samples,) + probs.shape)
    self.assertTrue(np.all(samples >= 0))
    self.assertTrue(np.all(samples <= total_count))
    self.assertion_fn(rtol=0.1)(np.mean(samples, axis=0), dist.mean())
    self.assertion_fn(rtol=0.1)(np.var(samples, axis=0), dist.variance())

  @chex.all_variants
  @parameterized.named_parameters(
      (
          '1d logits, int value',
          {'total_count': 10, 'logits': [0.0, 0.5, -0.5]},
          1,
      ),
      ('1d probs, int value', {'total_count': 10, 'probs': [0.3, 0.2, 0.5]}, 1),
      (
          '1d logits, 1d value',
          {'total_count': 10, 'logits': [0.0, 0.5, -0.5]},
          [1, 0, 8],
      ),
      (
          '1d probs, 1d value',
          {'total_count': 10, 'probs': [0.3, 0.2, 0.5]},
          [1, 0, 8],
      ),
      (
          'broadcast n, 1d value',
          {'total_count': [5, 10], 'probs': [0.1, 0.8]},
          [2, 7],
      ),
      (
          'edge cases with logits',
          {'total_count': 5, 'logits': [-np.inf, -np.inf, np.inf, np.inf]},
          [0, 1, 0, 5],
      ),
      (
          'edge cases with probs',
          {'total_count': 5, 'probs': [0.0, 0.0, 1.0, 1.0]},
          [0, 1, 0, 5],
      ),
      ('invalid value > n', {'total_count': 5, 'probs': [0.5]}, 6),
      ('invalid value < 0', {'total_count': 5, 'probs': [0.5]}, -1),
      ('invalid float value', {'total_count': 5, 'probs': [0.5]}, 2.5),
  )
  def test_method_with_value(self, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = jnp.asarray(value)
    for method in [
        'prob',
        'log_prob',
        'cdf',
        'log_cdf',
        'survival_function',
        'log_survival_function',
    ]:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2),
        )

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      (
          'from logits',
          {'total_count': 10, 'logits': [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
      ),
      (
          'from probs',
          {'total_count': 10, 'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]},
      ),
      (
          'broadcast n',
          {'total_count': 5, 'probs': [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]},
      ),
  )
  def test_method(self, distr_params):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    for method in ['entropy', 'mode', 'mean', 'variance', 'stddev']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(),
            assertion_fn=self.assertion_fn(rtol=1e-2),
        )

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
            'total_count': jnp.array([5, 10]),
            'probs': jnp.asarray([[0.1, 0.5], [0.2, 0.8]]),
        },
        dist2_kwargs={
            'total_count': jnp.array([5, 10]),
            'logits': jnp.asarray([0.0, -0.1]),
        },
        assertion_fn=self.assertion_fn(rtol=1e-2),
    )

  def test_kl_divergence_different_n(self):
    # KL should be inf if total_count is different.
    # This test is run in eager mode only.
    dist1_kwargs = {'total_count': jnp.array([5, 10]), 'probs': [0.1, 0.2]}
    dist2_kwargs = {'total_count': jnp.array([5, 11]), 'probs': [0.1, 0.2]}
    dist1 = self.distrax_cls(**dist1_kwargs)
    dist2 = self.distrax_cls(**dist2_kwargs)

    kl_val = dist1.kl_divergence(dist2)

    self.assertEqual(kl_val.shape, (2,))
    self.assertFalse(np.isinf(kl_val[0]))
    self.assertTrue(np.isinf(kl_val[1]))
    self.assertFalse(np.any(np.isnan(kl_val)))

  def test_jittable(self):
    dist_kwargs = {'total_count': 5.0, 'logits': [0.0, 4.0, -1.0, 4.0]}
    super()._test_jittable(
        dist_kwargs=dist_kwargs, assertion_fn=self.assertion_fn(rtol=1e-3)
    )

  @parameterized.named_parameters(
      ('single element, from probs', 2, True),
      ('single element, from logits', 2, False),
      ('range, from probs', slice(-1), True),
      ('range, from logits', slice(-1), False),
  )
  def test_slice(self, slice_, from_probs):
    rng = np.random.default_rng(42)
    total_count = rng.integers(low=1, high=100, size=(3, 4, 5))
    probs = rng.uniform(size=(3, 4, 5))
    logits = sp_special.logit(probs)

    dist_kwargs = {'total_count': total_count}
    if from_probs:
      dist_kwargs['probs'] = probs
    else:
      dist_kwargs['logits'] = logits

    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist[slice_].logits, logits[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].probs, probs[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].total_count, total_count[slice_])


if __name__ == '__main__':
  absltest.main()
