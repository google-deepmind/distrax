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
"""Tests for `categorical_uniform.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import categorical_uniform
import jax
import jax.numpy as jnp
import numpy as np

_NAMED_PARAMETERS = (
    dict(
        testcase_name='scalar',
        low=0.,
        high=1.,
        logits=np.zeros((7,)),
        num_bins=7,
        target_event_shape=(),
        target_sample_shape=(),
        target_batch_shape=(),
        target_low=np.zeros(()),
        target_high=np.ones(()),
        target_logits=np.zeros((7,)),
        target_entropy=np.float64(0.0),
        target_mean=np.float32(0.5),
        target_variance=np.float32(1/12),
    ),
    dict(
        testcase_name='one_dimensional',
        low=np.zeros((2,)),
        high=np.ones((2,)),
        logits=np.zeros((2, 7)),
        num_bins=7,
        target_event_shape=(),
        target_sample_shape=(2,),
        target_batch_shape=(2,),
        target_low=np.zeros((2,)),
        target_high=np.ones((2,)),
        target_logits=np.zeros((2, 7)),
        target_entropy=np.full((2,), 0.0, dtype=np.float64),
        target_mean=np.full((2,), 0.5),
        target_variance=np.full((2,), 1/12),
    ),
    dict(
        testcase_name='two_dimensional',
        low=np.zeros((2, 3)),
        high=np.ones((2, 3)),
        logits=np.zeros((2, 3, 7)),
        num_bins=7,
        target_event_shape=(),
        target_sample_shape=(2, 3),
        target_batch_shape=(2, 3),
        target_low=np.zeros((2, 3)),
        target_high=np.ones((2, 3)),
        target_logits=np.zeros((2, 3, 7)),
        target_entropy=np.full((2, 3), 0.0, dtype=np.float64),
        target_mean=np.full((2, 3), 0.5),
        target_variance=np.full((2, 3), 1/12),
    ),
    dict(
        testcase_name='broadcasted_low',
        low=0.,
        high=np.ones((2, 3)),
        logits=np.zeros((2, 3, 7)),
        num_bins=7,
        target_event_shape=(),
        target_sample_shape=(2, 3),
        target_batch_shape=(2, 3),
        target_low=np.zeros((2, 3)),
        target_high=np.ones((2, 3)),
        target_logits=np.zeros((2, 3, 7)),
        target_entropy=np.full((2, 3), 0.0, dtype=np.float64),
        target_mean=np.full((2, 3), 0.5),
        target_variance=np.full((2, 3), 1/12),
    ),
    dict(
        testcase_name='broadcasted_high',
        low=np.zeros((2, 3)),
        high=1.,
        logits=np.zeros((2, 3, 7)),
        num_bins=7,
        target_event_shape=(),
        target_sample_shape=(2, 3),
        target_batch_shape=(2, 3),
        target_low=np.zeros((2, 3)),
        target_high=np.ones((2, 3)),
        target_logits=np.zeros((2, 3, 7)),
        target_entropy=np.full((2, 3), 0.0, dtype=np.float64),
        target_mean=np.full((2, 3), 0.5),
        target_variance=np.full((2, 3), 1/12),
    ),
    dict(
        testcase_name='broadcasted_logits',
        low=np.zeros((2, 3)),
        high=np.ones((2, 3)),
        logits=np.zeros((7,)),
        num_bins=7,
        target_event_shape=(),
        target_sample_shape=(2, 3),
        target_batch_shape=(2, 3),
        target_low=np.zeros((2, 3)),
        target_high=np.ones((2, 3)),
        target_logits=np.zeros((2, 3, 7)),
        target_entropy=np.full((2, 3), 0.0, dtype=np.float64),
        target_mean=np.full((2, 3), 0.5),
        target_variance=np.full((2, 3), 1/12),
    ),
)


class CategoricalUniformTest(parameterized.TestCase):

  def test_raises_on_wrong_logits(self):
    with self.assertRaises(ValueError):
      categorical_uniform.CategoricalUniform(low=0., high=1., logits=0.)

  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_batch_shape(self, *, low, high, logits, target_batch_shape, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    self.assertEqual(distribution.batch_shape, target_batch_shape)

  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_event_shape(self, *, low, high, logits, target_event_shape, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    self.assertEqual(distribution.event_shape, target_event_shape)

  @chex.all_variants
  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_sample_shape(self, *, low, high, logits, target_sample_shape, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    sample = self.variant(distribution.sample)(seed=jax.random.PRNGKey(42))
    self.assertEqual(sample.shape, target_sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_sample_gradients(self, *, low, high, logits, **_):
    def summed_samples_fn(params):
      distribution = categorical_uniform.CategoricalUniform(
          low=params[0], high=params[1], logits=params[2])
      sample = distribution.sample(seed=jax.random.PRNGKey(42))
      return sample.sum()
    grad_fn = self.variant(jax.grad(summed_samples_fn))
    grad_low, grad_high, grad_logits = grad_fn(
        (jnp.float32(low), jnp.float32(high), jnp.float32(logits)))
    self.assertTrue(np.all(grad_low))  # Assert gradient is non-zero.
    self.assertTrue(np.all(grad_high))  # Assert gradient is non-zero.
    self.assertTrue(np.all(grad_logits))  # Assert gradient is non-zero.

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_entropy(self, *, low, high, logits, target_entropy, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    chex.assert_trees_all_close(
        self.variant(distribution.entropy)(),
        target_entropy,
        atol=1e-4,
        rtol=1e-4,
    )

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_mean(self, *, low, high, logits, target_mean, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    chex.assert_trees_all_close(
        self.variant(distribution.mean)(), target_mean, atol=1e-4, rtol=1e-4)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_variance(self, *, low, high, logits, target_variance, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    chex.assert_trees_all_close(
        self.variant(distribution.variance)(),
        target_variance,
        atol=1e-4,
        rtol=1e-4,
    )

  @chex.all_variants
  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_log_prob(self, *, low, high, logits, target_sample_shape, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    sample = jnp.full(target_sample_shape, 0.2, jnp.float32)
    log_prob = self.variant(distribution.log_prob)(sample)
    target_log_prob = jnp.zeros(target_sample_shape, jnp.float32)
    chex.assert_trees_all_close(log_prob, target_log_prob, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(*_NAMED_PARAMETERS)
  def test_attributes(
      self, *,
      low, high, logits, target_low, target_high, target_logits, **_,
  ):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    with self.subTest('low'):
      chex.assert_trees_all_equal(distribution.low, target_low)
    with self.subTest('high'):
      chex.assert_trees_all_equal(distribution.high, target_high)
    with self.subTest('logits'):
      chex.assert_trees_all_equal(distribution.logits, target_logits)

  @parameterized.named_parameters(
      # Remove the scalar parameterization because slice would be out of range.
      named_parameters for named_parameters in _NAMED_PARAMETERS
      if named_parameters['testcase_name'] != 'scalar'
  )
  def test_slice(self, low, high, logits, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    for name, key in (
        ('single_element', 1),
        ('range', slice(-1)),
        ('ellipsis', (Ellipsis, -1)),
    ):
      with self.subTest(name):
        chex.assert_trees_all_close(
            distribution[key].low,
            distribution.low[key],
            atol=1e-4,
            rtol=1e-4,
        )
        chex.assert_trees_all_close(
            distribution[key].high,
            distribution.high[key],
            atol=1e-4,
            rtol=1e-4,
        )
        dist_logits = distribution.logits
        chex.assert_trees_all_close(
            distribution[key].logits,
            dist_logits[key] if name != 'ellipsis' else dist_logits[..., -1, :],
            atol=1e-4,
            rtol=1e-4,
        )

  @chex.all_variants
  @parameterized.named_parameters(
      named_parameters for named_parameters in _NAMED_PARAMETERS
      if named_parameters['testcase_name'] == 'scalar'
  )
  def test_log_prob_outside_of_domain(
      self, *, low, high, logits, target_sample_shape, **_):
    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    with self.subTest('lower'):
      sample = jnp.full(target_sample_shape, -1, jnp.float32)
      log_prob = self.variant(distribution.log_prob)(sample)
      self.assertEqual(log_prob, -np.inf)
    with self.subTest('upper'):
      sample = jnp.full(target_sample_shape, +2, jnp.float32)
      log_prob = self.variant(distribution.log_prob)(sample)
      self.assertEqual(log_prob, -np.inf)

  @parameterized.named_parameters(
      named_parameters for named_parameters in _NAMED_PARAMETERS
      if named_parameters['testcase_name'] == 'two_dimensional'
  )
  def test_vmap_inputs(self, *, low, high, logits, target_sample_shape, **_):
    def log_prob_sum(distribution, sample):
      return distribution.log_prob(sample).sum()

    distribution = categorical_uniform.CategoricalUniform(
        low=low, high=high, logits=logits)
    sample = jnp.full(target_sample_shape, 0.2, jnp.float32)

    with self.subTest('no vmap'):
      actual = log_prob_sum(distribution, sample)
      expected = distribution.log_prob(sample).sum()
      chex.assert_trees_all_close(actual, expected, atol=1e-4, rtol=1e-4)

    with self.subTest('axis=0'):
      actual = jax.vmap(log_prob_sum, in_axes=0)(distribution, sample)
      expected = distribution.log_prob(sample).sum(axis=1)
      chex.assert_trees_all_close(actual, expected, atol=1e-4, rtol=1e-4)

    with self.subTest('axis=1'):
      actual = jax.vmap(log_prob_sum, in_axes=1)(distribution, sample)
      expected = distribution.log_prob(sample).sum(axis=0)
      chex.assert_trees_all_close(actual, expected, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      named_parameters for named_parameters in _NAMED_PARAMETERS
      if named_parameters['testcase_name'] == 'two_dimensional'
  )
  def test_vmap_outputs(self, *, low, high, logits, target_sample_shape, **_):
    def summed_distribution(low, high, logits):
      return categorical_uniform.CategoricalUniform(
          low=low.sum(keepdims=True),
          high=high.sum(keepdims=True),
          logits=logits.sum(keepdims=True),
      )

    actual = jax.vmap(summed_distribution)(low, high, logits)
    expected = categorical_uniform.CategoricalUniform(
        low=low.sum(axis=1, keepdims=True),
        high=high.sum(axis=1, keepdims=True),
        logits=logits.sum(axis=1, keepdims=True),
    )
    np.testing.assert_equal(actual.batch_shape, expected.batch_shape)
    np.testing.assert_equal(actual.event_shape, expected.event_shape)
    sample = jnp.full(target_sample_shape, 0.2, jnp.float32)
    chex.assert_trees_all_close(
        actual.log_prob(sample),
        expected.log_prob(sample),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_jitable(self):

    @jax.jit
    def jitted_function(event, dist):
      return dist.log_prob(event)

    dist = categorical_uniform.CategoricalUniform(
        low=0., high=1., logits=np.ones((7,)))
    event = dist.sample(seed=jax.random.PRNGKey(4242))
    log_prob = dist.log_prob(event)
    jitted_log_prob = jitted_function(event, dist)
    chex.assert_trees_all_close(jitted_log_prob, log_prob, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
