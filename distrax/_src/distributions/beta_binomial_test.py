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
      ('1d params', (4,), 10),
      ('2d params', (3, 4), 10),
      ('scalar params', (), 5),
  )
  def test_properties(self, shape, total_count):
    rng = np.random.default_rng(42)
    concentration1 = rng.uniform(size=shape) + 0.5
    concentration0 = rng.uniform(size=shape) + 0.5
    dist = self.distrax_cls(
        total_count=total_count,
        concentration1=concentration1,
        concentration0=concentration0,
    )

    self.assertion_fn(rtol=1e-3)(dist.total_count, total_count)
    self.assertion_fn(rtol=1e-3)(dist.concentration1, concentration1)
    self.assertion_fn(rtol=1e-3)(dist.concentration0, concentration0)

    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, shape)

  @parameterized.named_parameters(
      (
          'complex64 dtype',
          {
              'total_count': 10,
              'concentration1': 1.0,
              'concentration0': 1.0,
              'dtype': jnp.complex64,
          },
      ),
      (
          'complex128 dtype',
          {
              'total_count': 10,
              'concentration1': 1.0,
              'concentration0': 1.0,
              'dtype': jnp.complex128,
          },
      ),
  )
  def test_raises_on_invalid_inputs(self, dist_params):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      (
          '1d params, no shape',
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
          (),
      ),
      (
          '1d params, int shape',
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
          1,
      ),
      (
          '1d params, 2-tuple shape',
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
          (5, 4),
      ),
      (
          '2d params, no shape',
          {
              'total_count': 5.0,
              'concentration1': [[0.5, 1.0], [1.0, 0.5]],
              'concentration0': 1.0,
          },
          (),
      ),
      (
          'broadcast params',
          {
              'total_count': [5.0, 10.0],
              'concentration1': [[0.5, 1.0], [1.0, 0.5]],
              'concentration0': 1.0,
          },
          (3,),
      ),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    super()._test_sample_shape(
        dist_args=(), dist_kwargs=distr_params, sample_shape=sample_shape
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('sample, int16', 'sample', jnp.int16),
      ('sample, int32', 'sample', jnp.int32),
      ('sample, float16', 'sample', jnp.float16),
      ('sample, float32', 'sample', jnp.float32),
      ('sample_and_log_prob, int16', 'sample_and_log_prob', jnp.int16),
      ('sample_and_log_prob, int32', 'sample_and_log_prob', jnp.int32),
      ('sample_and_log_prob, float16', 'sample_and_log_prob', jnp.float16),
      ('sample_and_log_prob, float32', 'sample_and_log_prob', jnp.float32),
  )
  def test_sample_dtype(self, method, dtype):
    dist_params = {
        'total_count': 10.0,
        'concentration1': jnp.array([0.5, 1.0]),
        'concentration0': jnp.array([1.0, 0.5]),
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
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
          1.0,
      ),
      (
          '1d params, 1d value',
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
          np.array([1, 5], dtype=np.float32),
      ),
      (
          '1d params, 2d value',
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
          np.array([[1, 5], [3, 8]], dtype=np.float32),
      ),
  )
  def test_method_with_value(self, distr_params, value):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    value = jnp.asarray(value)
    for method in ['log_prob', 'prob']:
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
          {'total_count': 10.0, 'concentration1': 0.5, 'concentration0': 1.0},
      ),
      (
          '1d params',
          {
              'total_count': 10.0,
              'concentration1': [0.5, 1.0],
              'concentration0': [1.0, 0.5],
          },
      ),
      (
          '2d params',
          {
              'total_count': 5.0,
              'concentration1': [[0.5, 1.0], [1.0, 0.5]],
              'concentration0': 1.0,
          },
      ),
  )
  def test_method(self, distr_params):
    distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
    for method in ['mean', 'variance']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_kwargs=distr_params,
            call_args=(),
            assertion_fn=self.assertion_fn(rtol=1e-2),
        )

  def test_jittable(self):
    dist_args = (10.0, jnp.array([0.5, 1.0]), jnp.array([1.0, 0.5]))
    super()._test_jittable(
        dist_args=dist_args, assertion_fn=self.assertion_fn(rtol=1e-3)
    )

  @parameterized.named_parameters(
      ('single element', 1),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(42)
    total_count = np.array([[10, 20], [30, 40], [50, 60]])[..., np.newaxis]
    concentration1 = rng.uniform(size=(3, 2, 5)) + 0.5
    concentration0 = rng.uniform(size=(3, 2, 5)) + 0.5
    dist = self.distrax_cls(
        total_count=total_count,
        concentration1=concentration1,
        concentration0=concentration0,
    )

    # We test self._total_count because the property `total_count` is
    # broadcast to batch_shape.
    self.assertion_fn(rtol=1e-3)(dist[slice_]._total_count, total_count[slice_])
    self.assertion_fn(rtol=1e-3)(
        dist[slice_]._concentration1, concentration1[slice_]
    )
    self.assertion_fn(rtol=1e-3)(
        dist[slice_]._concentration0, concentration0[slice_]
    )


if __name__ == '__main__':
  absltest.main()
