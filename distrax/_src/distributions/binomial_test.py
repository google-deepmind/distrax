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
    self_p = np.asarray([0.2, 0.4, 0.6, 0.8])
    self.logits = sp_special.logit(self_p)
    self.total_count = 10

  @parameterized.named_parameters(
      ('1d probs', (4,), True),
      ('1d logits', (4,), False),
      ('2d probs', (3, 4), True),
      ('2d logits', (3, 4), False),
  )
  def test_properties(self, shape, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=shape)
    logits = sp_special.logit(probs)
    total_count = np.full(shape, self.total_count, dtype=int)

    dist_kwargs = (
        {'total_count': total_count, 'probs': probs}
        if from_probs
        else {'total_count': total_count, 'logits': logits}
    )

    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.logits, logits)
    self.assertion_fn(rtol=1e-3)(dist.probs, probs)
    self.assertion_fn(rtol=1e-3)(dist.total_count, total_count)
    self.assertEqual(dist.event_shape, ())

  @parameterized.named_parameters(
      ('1d probs', (4,), True),
      ('1d logits', (4,), False),
      ('2d probs', (3, 4), True),
      ('2d logits', (3, 4), False),
  )
  def test_mean(self, shape, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=shape)
    logits = sp_special.logit(probs)
    total_count = np.full(shape, self.total_count, dtype=int)

    dist_kwargs = (
        {'total_count': total_count, 'probs': probs}
        if from_probs
        else {'total_count': total_count, 'logits': logits}
    )

    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.mean(), total_count * probs)

  @parameterized.named_parameters(
      ('1d probs', (4,), True),
      ('1d logits', (4,), False),
      ('2d probs', (3, 4), True),
      ('2d logits', (3, 4), False),
  )
  def test_variance_and_stddev(self, shape, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=shape)
    logits = sp_special.logit(probs)
    total_count = np.full(shape, self.total_count, dtype=int)

    dist_kwargs = (
        {'total_count': total_count, 'probs': probs}
        if from_probs
        else {'total_count': total_count, 'logits': logits}
    )

    dist = self.distrax_cls(**dist_kwargs)
    expected_variance = total_count * probs * (1.0 - probs)
    self.assertion_fn(rtol=1e-3)(dist.variance(), expected_variance)
    self.assertion_fn(rtol=1e-3)(dist.stddev(), np.sqrt(expected_variance))

  @parameterized.named_parameters(
      ('1d probs', (4,), True),
      ('1d logits', (4,), False),
      ('2d probs', (3, 4), True),
      ('2d logits', (3, 4), False),
  )
  def test_mode(self, shape, from_probs):
    rng = np.random.default_rng(42)
    probs = rng.uniform(size=shape)
    logits = sp_special.logit(probs)
    total_count = np.full(shape, self.total_count, dtype=int)

    dist_kwargs = (
        {'total_count': total_count, 'probs': probs}
        if from_probs
        else {'total_count': total_count, 'logits': logits}
    )

    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.mode(), np.floor(total_count * probs))

  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64),
  )
  def test_median_not_implemented(self, dtype):
    logits = jnp.array(sp_special.logit(np.array([0.1, 0.5, 0.8]))).astype(
        dtype
    )
    total_count = jnp.array([5, 10, 8]).astype(int)
    dist = self.distrax_cls(total_count=total_count, logits=logits)
    with self.assertRaises(NotImplementedError):
      dist.median()

  def test_jittable(self):
    dist_kwargs = {'total_count': 10, 'probs': 0.1, 'dtype': jnp.int32}
    super()._test_jittable(dist_args=(), dist_kwargs=dist_kwargs)

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, from probs', 'sample', True),
      ('sample, from logits', 'sample', False),
      ('sample_and_log_prob, from probs', 'sample_and_log_prob', True),
      ('sample_and_log_prob, from logits', 'sample_and_log_prob', False),
  )
  def test_sample_values(self, method, from_probs):
    probs = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    logits = sp_special.logit(probs)
    total_count = 10
    n_samples = 100000
    dist_kwargs = (
        {'total_count': total_count, 'probs': probs}
        if from_probs
        else {'total_count': total_count, 'logits': logits}
    )
    dist = self.distrax_cls(**dist_kwargs)
    sample_fn = self.variant(
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples)
    )
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.shape, (n_samples,) + probs.shape)
    self.assertion_fn(rtol=0.1)(np.mean(samples, axis=0), dist.mean())
    self.assertion_fn(rtol=0.1)(np.std(samples, axis=0), dist.stddev())

  def test_slice(self):
    total_count = jnp.array([10, 10, 10])
    logits = jnp.array([0.1, 0.2, 0.3])
    dist = self.distrax_cls(total_count=total_count, logits=logits)
    self.assertIsInstance(dist[0], self.distrax_cls)
    self.assertEqual(dist[0].batch_shape, ())
    self.assertEqual(dist[:1].batch_shape, (1,))
    self.assertion_fn()(dist[0].total_count, 10)
    self.assertion_fn()(dist[0].logits, 0.1)


if __name__ == '__main__':
  absltest.main()
