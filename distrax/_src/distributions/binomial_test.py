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


class BinomialTest(equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(binomial.Binomial)

  @parameterized.named_parameters(
      ('1d logits', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}),
      ('1d probs', (10,), {'probs': np.array([0.1, 0.5, 0.9])}),
      (
          '2d logits',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
      ),
      (
          '2d probs',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
      (
          'broadcasted n',
          (10,),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
  )
  def test_properties(self, n, dist_params):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    tfp_dist = self.tfp_cls(total_count=n, **dist_kwargs)

    self.assertion_fn(rtol=1e-3)(dist.logits, tfp_dist.logits)
    self.assertion_fn(rtol=1e-3)(dist.probs, tfp_dist.probs)
    self.assertion_fn()(dist.total_count, tfp_dist.total_count)
    self.assertEqual(dist.event_shape, tfp_dist.event_shape)
    self.assertEqual(dist.batch_shape, tfp_dist.batch_shape)
    # Check that n is broadcast correctly.
    self.assertEqual(dist.total_count.shape, dist.batch_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}),
      ('1d probs', (10,), {'probs': np.array([0.1, 0.5, 0.9])}),
      (
          '2d logits',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
      ),
      (
          '2d probs',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
  )
  def test_log_prob(self, n, dist_params):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    n_max = np.max(np.asarray(n))
    values = np.arange(0, n_max + 1).astype(np.float32)
    values = values.reshape(values.shape + (1,) * len(dist.batch_shape))

    super()._test_log_prob(
        dist_args=(n,), dist_kwargs=dist_kwargs, value=values
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}),
      ('1d probs', (10,), {'probs': np.array([0.1, 0.5, 0.9])}),
      (
          '2d logits',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
      ),
      (
          '2d probs',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
  )
  def test_prob(self, n, dist_params):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    n_max = np.max(np.asarray(n))
    values = np.arange(0, n_max + 1).astype(np.float32)
    values = values.reshape(values.shape + (1,) * len(dist.batch_shape))

    super()._test_prob(dist_args=(n,), dist_kwargs=dist_kwargs, value=values)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}),
      ('1d probs', (10,), {'probs': np.array([0.1, 0.5, 0.9])}),
      (
          '2d logits',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
      ),
      (
          '2d probs',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
  )
  def test_cdf(self, n, dist_params):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    n_max = np.max(np.asarray(n))
    values = np.arange(-1, n_max + 2).astype(np.float32)
    values = values.reshape(values.shape + (1,) * len(dist.batch_shape))

    super()._test_cdf(dist_args=(n,), dist_kwargs=dist_kwargs, value=values)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, 1 sample', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}, 1),
      ('1d probs, 1 sample', (10,), {'probs': np.array([0.1, 0.5, 0.9])}, 1),
      (
          '2d logits, 1 sample',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
          1,
      ),
      (
          '2d probs, 1 sample',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
          1,
      ),
      (
          '1d logits, 5 samples',
          (10,),
          {'logits': np.array([-1.0, 0.0, 1.0])},
          5,
      ),
      ('1d probs, 5 samples', (10,), {'probs': np.array([0.1, 0.5, 0.9])}, 5),
      (
          '2d logits, 5 samples',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
          5,
      ),
      (
          '2d probs, 5 samples',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
          5,
      ),
  )
  def test_sample_shape(self, n, dist_params, n_samples):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    super()._test_sample_shape(
        dist_args=(n,), dist_kwargs=dist_kwargs, sample_shape=n_samples
    )
    super()._test_sample_shape(
        dist_args=(n,), dist_kwargs=dist_kwargs, sample_shape=(n_samples, 2)
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}),
      ('1d probs', (10,), {'probs': np.array([0.1, 0.5, 0.9])}),
      (
          '2d logits',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
      ),
      (
          '2d probs',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
  )
  def test_sample_values(self, n, dist_params):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)

    sample_fn = self.variant(  # pytype: disable=attribute-error
        lambda key: dist.sample(seed=key, sample_shape=(1_000,))
    )
    samples = sample_fn(self.key)

    self.assertIs(samples.dtype, jnp.dtype(dist.dtype))
    self.assertTrue(jnp.all(samples >= 0))

    # Check max value against broadcasted total_count
    broadcasted_total_count = jnp.broadcast_to(
        dist.total_count, dist.batch_shape
    )
    self.assertTrue(jnp.all(samples <= broadcasted_total_count))

    # Check that the mean of samples is close to the analytic mean.
    self.assertion_fn(rtol=0.3)(  # Loosen tolerance for fewer samples
        jnp.mean(samples, axis=0), dist.mean()
    )

  @chex.all_variants
  @parameterized.named_parameters(
      (
          '1d logits, float32',
          (10,),
          {'logits': np.array([-1.0, 0.0, 1.0]), 'dtype': jnp.float32},
      ),
      (
          '1d probs, float32',
          (10,),
          {'probs': np.array([0.1, 0.5, 0.9]), 'dtype': jnp.float32},
      ),
      (
          '2d logits, float32',
          (np.array([[10], [20]]),),
          {
              'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]]),
              'dtype': jnp.float32,
          },
      ),
      (
          '2d probs, float32',
          (np.array([[10], [20]]),),
          {
              'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]]),
              'dtype': jnp.float32,
          },
      ),
  )
  def test_sample_dtype(self, n, dist_params, n_samples=10):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    samples = dist.sample(seed=self.key, sample_shape=n_samples)
    self.assertEqual(samples.shape, (n_samples,) + dist.batch_shape)
    self.assertEqual(samples.dtype, dist.dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}),
      ('1d probs', (10,), {'probs': np.array([0.1, 0.5, 0.9])}),
      (
          '2d logits',
          (np.array([[10], [20]]),),
          {'logits': np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])},
      ),
      (
          '2d probs',
          (np.array([[10], [20]]),),
          {'probs': np.array([[0.1, 0.5, 0.9], [0.3, 0.7, 0.8]])},
      ),
  )
  def test_jittable(self, n, dist_params):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    super()._test_jittable(dist_args=(n,), dist_kwargs=dist_kwargs)

  @parameterized.named_parameters(
      ('no args', (10,), {}, ValueError),
      ('both args', (10,), {'logits': 0.1, 'probs': 0.8}, ValueError),
      ('wrong dtype', (10,), {'logits': 0.1, 'dtype': jnp.uint8}, ValueError),
      ('negative n', (-10,), {'logits': 0.1}, ValueError),
  )
  def test_raises_on_invalid_inputs(self, n, dist_params, error):
    with self.assertRaises(error):
      self.distrax_cls(total_count=n, **dist_params)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, int', (10,), {'logits': np.array([-1.0, 0.0, 1.0])}, int),
      ('1d probs, int', (10,), {'probs': np.array([0.1, 0.5, 0.9])}, int),
      (
          '1d logits, float',
          (10,),
          {'logits': np.array([-1.0, 0.0, 1.0]), 'dtype': jnp.float32},
          jnp.float32,
      ),
      (
          '1d probs, float',
          (10,),
          {'probs': np.array([0.1, 0.5, 0.9]), 'dtype': jnp.float32},
          jnp.float32,
      ),
  )
  def test_default_dtype(self, n, dist_params, expected_dtype):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    self.assertEqual(dist.dtype, expected_dtype)
    self.assertEqual(dist.mode().dtype, expected_dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('int slice', (10,), {'logits': np.array([-1.0, 0.0, 1.0, 0.5])}, 1),
      (
          'slice',
          (10,),
          {'probs': np.array([[0.1, 0.5], [0.2, 0.3]])},
          (slice(None), 0),
      ),
      (
          'fancy index',
          (np.array([10, 20, 30, 40]),),
          {'logits': np.array([-1.0, 0.0, 1.0, 0.5])},
          np.array([0, 2]),
      ),
  )
  def test_slicing(self, n, dist_params, index):
    dist_kwargs = {k: jnp.asarray(v) for k, v in dist_params.items()}
    dist = self.distrax_cls(total_count=n, **dist_kwargs)
    sliced_dist = dist[index]

    tfp_dist = self.tfp_cls(total_count=n, **dist_kwargs)
    sliced_tfp_dist = tfp_dist[index]

    self.assertion_fn()(sliced_dist.logits, sliced_tfp_dist.logits)
    self.assertion_fn()(sliced_dist.probs, sliced_tfp_dist.probs)
    self.assertion_fn()(sliced_dist.total_count, sliced_tfp_dist.total_count)

  def test_entropy_not_implemented(self):
    dist = self.distrax_cls(total_count=10, probs=0.1)
    with self.assertRaises(NotImplementedError):
      dist.entropy()


if __name__ == '__main__':
  absltest.main()
