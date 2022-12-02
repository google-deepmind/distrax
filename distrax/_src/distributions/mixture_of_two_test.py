# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `mixture_of_two.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import mixture_of_two
from distrax._src.utils import equivalence
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


BATCH_SIZE = 5
PROPORTION = 0.3


class MixtureOfTwoTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(mixture_of_two.MixtureOfTwo)
    components = self._get_components(hk.PRNGSequence(0))
    self.component_a = components[0]
    self.component_b = components[1]
    self.tfp_mixture = components[2]

  def assertion_fn(self, rtol: float = 1e-3):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  def _get_components(self, rng_seq):
    loc = jax.random.normal(next(rng_seq), (BATCH_SIZE, 2))
    scale = jax.nn.sigmoid(jax.random.normal(next(rng_seq), (BATCH_SIZE, 2)))
    component_a = tfp.distributions.Normal(loc=loc[:, 0], scale=scale[:, 0])
    component_b = tfp.distributions.Normal(loc=loc[:, 1], scale=scale[:, 1])
    full_proportion = jnp.full([BATCH_SIZE], PROPORTION)
    tfp_mixture = tfp.distributions.MixtureSameFamily(
        tfp.distributions.Categorical(probs=jnp.stack(
            [full_proportion, 1 - full_proportion], axis=-1)),
        components_distribution=tfp.distributions.Normal(loc, scale))
    return component_a, component_b, tfp_mixture

  def test_mixture_methods(self):
    rng_seq = hk.PRNGSequence(0)
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)

    sample_shape = (8, 1024)
    sample = self.tfp_mixture.sample(
        sample_shape=sample_shape, seed=next(rng_seq))
    other_sample = mix.sample(sample_shape=sample_shape, seed=next(rng_seq))
    chex.assert_equal_shape([sample, other_sample])
    np.testing.assert_allclose(
        sample.mean(axis=[0, 1]), other_sample.mean(axis=[0, 1]), atol=1e-1)
    np.testing.assert_allclose(
        sample.std(axis=[0, 1]), other_sample.std(axis=[0, 1]), atol=1e-1)
    np.testing.assert_allclose(
        self.tfp_mixture.log_prob(sample),
        mix.log_prob(sample), atol=1e-3)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('empty shape', ()),
      ('int shape', 10),
      ('2-tuple shape', (10, 20)),
  )
  def test_sample_and_log_prob(self, sample_shape):
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    expected_event = mix.sample(
        seed=jax.random.PRNGKey(42), sample_shape=sample_shape)
    expected_log_prob = mix.log_prob(expected_event)
    event, log_prob = self.variant(
        mix.sample_and_log_prob, static_argnames='sample_shape')(
            seed=jax.random.PRNGKey(42), sample_shape=sample_shape)

    np.testing.assert_allclose(expected_log_prob, log_prob, atol=1e-3)
    np.testing.assert_allclose(expected_event, event, atol=1e-3)

  def test_jitable(self):
    @jax.jit
    def jitted_function(event, dist):
      return dist.log_prob(event)

    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    event = mix.sample(seed=jax.random.PRNGKey(4242))
    log_prob = mix.log_prob(event)
    jitted_log_prob = jitted_function(event, mix)

    chex.assert_trees_all_close(
        jitted_log_prob, log_prob, atol=1e-4, rtol=1e-4)

  def test_prob_a(self):
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    self.assertEqual(mix.prob_a, PROPORTION)

  def test_prob_b(self):
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    self.assertEqual(mix.prob_b, 1. - PROPORTION)

  def test_batch_shape(self):
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    self.assertEqual(mix.batch_shape, (BATCH_SIZE,))
    self.assertEqual(mix.batch_shape, (BATCH_SIZE,))

  def test_event_shape(self):
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    self.assertEqual(mix.event_shape, ())
    self.assertEqual(mix.event_shape, ())

  @parameterized.named_parameters(
      ('single element', 1, ()),
      ('range', slice(-1), (4,)),
  )
  def test_slice(self, slice_, expected_batch_shape):
    mix = self.distrax_cls(PROPORTION, self.component_a, self.component_b)
    sliced_dist = mix[slice_]
    self.assertEqual(sliced_dist.batch_shape, expected_batch_shape)
    self.assertEqual(sliced_dist.event_shape, mix.event_shape)
    self.assertIsInstance(sliced_dist, self.distrax_cls)

  def test_invalid_parameters(self):
    rng_seq = hk.PRNGSequence(0)
    loc = jax.random.normal(next(rng_seq), (BATCH_SIZE,))
    scale = jax.nn.sigmoid(jax.random.normal(next(rng_seq), (BATCH_SIZE,)))
    concentration = jax.random.normal(next(rng_seq), (BATCH_SIZE,))
    with self.assertRaisesRegex(ValueError, 'must have the same event shape'):
      component_a = tfp.distributions.Normal(loc=loc, scale=scale)
      component_b = tfp.distributions.Dirichlet(concentration=concentration)
      self.distrax_cls(PROPORTION, component_a, component_b)

    with self.assertRaisesRegex(ValueError, 'must have the same batch shape'):
      component_a = tfp.distributions.Normal(loc=loc, scale=scale)
      component_b = tfp.distributions.Normal(loc=loc[1:], scale=scale[1:])
      self.distrax_cls(PROPORTION, component_a, component_b)

    with self.assertRaisesRegex(ValueError, 'must have the same dtype'):
      component_a = tfp.distributions.Normal(
          loc=loc.astype(jnp.float16), scale=scale.astype(jnp.float16))
      component_b = tfp.distributions.Normal(loc=loc, scale=scale)
      self.distrax_cls(PROPORTION, component_a, component_b)


if __name__ == '__main__':
  absltest.main()
