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
from distrax._src.distributions import beta_binomial
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


class BetaBinomialTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(beta_binomial.BetaBinomial)

  @parameterized.named_parameters(
      ('1d params', (4,), (4,), (4,)),
      ('broadcast 1d total_count', (1,), (4,), (4,)),
      ('broadcast 1d conc1', (4,), (1,), (4,)),
      ('broadcast 1d conc0', (4,), (4,), (1,)),
      ('broadcast all', (1,), (1,), (1,)),
      ('2d params', (3, 4), (3, 4), (3, 4)),
      ('broadcast 2d total_count', (1, 1), (3, 4), (3, 4)),
      ('broadcast 2d conc1', (3, 4), (1, 1), (3, 4)),
      ('broadcast 2d conc0', (3, 4), (3, 4), (1, 1)),
      ('mixed dims', (3, 1), (1, 4), (3, 4)),
  )
  def test_properties(self, n_shape, c1_shape, c0_shape):
    rng = np.random.default_rng(42)
    n = np.abs(rng.normal(size=n_shape)).astype(np.float32) * 10 + 1
    c1 = np.abs(rng.normal(size=c1_shape)).astype(np.float32) + 0.1
    c0 = np.abs(rng.normal(size=c0_shape)).astype(np.float32) + 0.1
    n = np.round(n)  # total_count should be integers

    dist = self.distrax_cls(total_count=n, concentration1=c1, concentration0=c0)

    # TFP broadcasting
    expected_batch_shape = np.broadcast_shapes(n_shape, c1_shape, c0_shape)

    self.assertion_fn(rtol=1e-3)(dist.total_count, n)
    self.assertion_fn(rtol=1e-3)(dist.concentration1, c1)
    self.assertion_fn(rtol=1e-3)(dist.concentration0, c0)
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, expected_batch_shape)

  @parameterized.named_parameters(
      (
          'complex64 dtype',
          {
              'total_count': 10.0,
              'concentration1': 1.0,
              'concentration0': 1.0,
              'dtype': jnp.complex64,
          },
      ),
      (
          'complex128 dtype',
          {
              'total_count': 10.0,
              'concentration1': 1.0,
              'concentration0': 1.0,
              'dtype': jnp.complex128,
          },
      ),
  )
  def test_raises_on_invalid_inputs(self, dist_params):
    # Note: distrax doesn't typically validate parameters (e.g. positivity)
    # at initialization, but rather during ops (log_prob, sample).
    # We test invalid dtype as in Bernoulli.
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      (
          '1d params, no shape',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [1.0, 2.0],
              'concentration0': [3.0, 4.0],
          },
          (),
      ),
      (
          '1d params, int shape',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [1.0, 2.0],
              'concentration0': [3.0, 4.0],
          },
          1,
      ),
      (
          '1d params, 1-tuple shape',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [1.0, 2.0],
              'concentration0': [3.0, 4.0],
          },
          (1,),
      ),
      (
          '1d params, 2-tuple shape',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [1.0, 2.0],
              'concentration0': [3.0, 4.0],
          },
          (5, 4),
      ),
      (
          'broadcast params, 2-tuple shape',
          {
              'total_count': 10.0,
              'concentration1': [1.0, 2.0],
              'concentration0': 3.0,
          },
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
      ('sample', 'sample'),
      ('sample_and_log_prob', 'sample_and_log_prob'),
  )
  def test_sample_values(self, method):
    n = np.array([5.0, 10.0])
    c1 = np.array([2.0, 4.0])
    c0 = np.array([3.0, 1.0])
    dist = self.distrax_cls(total_count=n, concentration1=c1, concentration0=c0)

    n_samples = 100000
    sample_fn = self.variant(
        lambda key: getattr(dist, method)(seed=key, sample_shape=n_samples)
    )
    samples = sample_fn(self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples

    self.assertEqual(samples.shape, (n_samples,) + n.shape)

    # Check samples are in [0, n]
    self.assertTrue(jnp.all(samples >= 0))
    self.assertTrue(jnp.all(samples <= jnp.broadcast_to(n, samples.shape)))

    # Check mean and variance
    self.assertion_fn(rtol=0.1)(jnp.mean(samples, axis=0), dist.mean())
    self.assertion_fn(rtol=0.1)(jnp.var(samples, axis=0), dist.variance())

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, int', 'sample', jnp.int32),
      ('sample, float', 'sample', jnp.float32),
      ('sample_and_log_prob, int', 'sample_and_log_prob', jnp.int32),
      ('sample_and_log_prob, float', 'sample_and_log_prob', jnp.float32),
  )
  def test_sample_dtype(self, method, dtype):
    dist_params = {
        'total_count': 10.0,
        'concentration1': 2.0,
        'concentration0': 3.0,
        'dtype': dtype,
    }
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(getattr(dist, method))(seed=self.key)
    samples = samples[0] if method == 'sample_and_log_prob' else samples
    self.assertEqual(samples.dtype, dist.dtype)
    self.assertEqual(samples.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      (
          '1d params, int value',
          {'total_count': 10.0, 'concentration1': 2.0, 'concentration0': 3.0},
          5,
      ),
      (
          '1d params, 1d value',
          {'total_count': 10.0, 'concentration1': 2.0, 'concentration0': 3.0},
          [1, 5, 10],
      ),
      (
          '1d params, 2d value',
          {'total_count': 10.0, 'concentration1': 2.0, 'concentration0': 3.0},
          [[1, 2], [8, 9]],
      ),
      (
          '2d params, 1d value',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [2.0, 4.0],
              'concentration0': [3.0, 5.0],
          },
          [5, 10],
      ),
      (
          '2d params, 2d value',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [2.0, 4.0],
              'concentration0': [3.0, 5.0],
          },
          [[1, 2], [8, 15]],
      ),
  )
  def test_method_with_value(self, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    # Cast value to float32 to avoid TFP dtype incompatibility
    value = jnp.asarray(value, dtype=jnp.float32)
    for method in ['prob', 'log_prob']:
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
          'scalar params',
          {'total_count': 10.0, 'concentration1': 2.0, 'concentration0': 3.0},
      ),
      (
          '1d params',
          {
              'total_count': [10.0, 20.0],
              'concentration1': [2.0, 4.0],
              'concentration0': [3.0, 5.0],
          },
      ),
      (
          '2d params',
          {
              'total_count': [[10.0]],
              'concentration1': [[2.0, 4.0]],
              'concentration0': 3.0,
          },
      ),
  )
  def test_method(self, distr_params):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    for method in ['mean', 'variance', 'stddev']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(),
            assertion_fn=self.assertion_fn(rtol=1e-2),
        )

  def test_jittable(self):
    dist_kwargs = {
        'total_count': 10.0,
        'concentration1': 1.0,
        'concentration0': 1.0,
    }
    # call_args is not a valid argument for the base class _test_jittable
    super()._test_jittable(
        dist_kwargs=dist_kwargs, assertion_fn=self.assertion_fn(rtol=1e-3)
    )

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    n = jnp.array(np.random.uniform(5, 15, size=(3, 4, 5)), dtype=jnp.float32)
    c1 = jnp.array(np.random.uniform(1, 5, size=(3, 4, 5)), dtype=jnp.float32)
    c0 = jnp.array(np.random.uniform(1, 5, size=(3, 4, 5)), dtype=jnp.float32)
    n = jnp.round(n)

    dist = self.distrax_cls(total_count=n, concentration1=c1, concentration0=c0)

    self.assertion_fn(rtol=1e-3)(dist[slice_].total_count, n[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].concentration1, c1[slice_])
    self.assertion_fn(rtol=1e-3)(dist[slice_].concentration0, c0[slice_])


if __name__ == '__main__':
  absltest.main()
