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
"""Tests for `mixture_of_two.py`."""

from absl.testing import absltest

import chex
from distrax._src.distributions import mixture_of_two
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


BATCH_SIZE = 5
PROPORTION = 0.3


class MixtureOfTwoTest(absltest.TestCase):

  def _get_components(self, rng_seq):
    loc = jax.random.normal(next(rng_seq), (BATCH_SIZE, 2))
    scale = jax.nn.sigmoid(jax.random.normal(next(rng_seq), (BATCH_SIZE, 2)))
    component1 = tfp.distributions.Normal(loc=loc[:, 0], scale=scale[:, 0])
    component2 = tfp.distributions.Normal(loc=loc[:, 1], scale=scale[:, 1])
    full_proportion = jnp.full([BATCH_SIZE], PROPORTION)
    tfp_mixture = tfp.distributions.MixtureSameFamily(
        tfp.distributions.Categorical(probs=jnp.stack(
            [full_proportion, 1 - full_proportion], axis=-1)),
        components_distribution=tfp.distributions.Normal(loc, scale))
    return component1, component2, tfp_mixture

  def test_mixture_methods(self):
    rng_seq = hk.PRNGSequence(0)
    component1, component2, tfp_mixture = self._get_components(rng_seq)
    mix = mixture_of_two.MixtureOfTwo(PROPORTION, component1, component2)

    sample_shape = (8, 1024)
    sample = tfp_mixture.sample(sample_shape=sample_shape, seed=next(rng_seq))
    other_sample = mix.sample(sample_shape=sample_shape, seed=next(rng_seq))
    chex.assert_equal_shape([sample, other_sample])
    np.testing.assert_allclose(
        sample.mean(axis=[0, 1]), other_sample.mean(axis=[0, 1]), atol=1e-1)
    np.testing.assert_allclose(
        sample.std(axis=[0, 1]), other_sample.std(axis=[0, 1]), atol=1e-1)
    np.testing.assert_allclose(
        tfp_mixture.log_prob(sample),
        mix.log_prob(sample), atol=1e-3)

  def test_jitable(self):

    @jax.jit
    def jitted_function(event, dist):
      return dist.log_prob(event)

    component1, component2, _ = self._get_components(hk.PRNGSequence(0))
    mix = mixture_of_two.MixtureOfTwo(PROPORTION, component1, component2)
    event = mix.sample(seed=jax.random.PRNGKey(4242))
    log_prob = mix.log_prob(event)
    jitted_log_prob = jitted_function(event, mix)

    chex.assert_trees_all_close(
        jitted_log_prob, log_prob, atol=1e-4, rtol=1e-4)

  def test_p_a(self):
    component1, component2, _ = self._get_components(hk.PRNGSequence(0))
    mix = mixture_of_two.MixtureOfTwo(PROPORTION, component1, component2)
    self.assertEqual(mix.p_a, PROPORTION)

  def test_p_b(self):
    component1, component2, _ = self._get_components(hk.PRNGSequence(0))
    mix = mixture_of_two.MixtureOfTwo(PROPORTION, component1, component2)
    self.assertEqual(mix.p_b, 1. - PROPORTION)

  def test_batch_shape(self):
    component1, component2, _ = self._get_components(hk.PRNGSequence(0))
    mix = mixture_of_two.MixtureOfTwo(PROPORTION, component1, component2)
    self.assertEqual(mix.batch_shape, (BATCH_SIZE,))
    self.assertEqual(mix.batch_shape, (BATCH_SIZE,))

  def test_event_shape(self):
    component1, component2, _ = self._get_components(hk.PRNGSequence(0))
    mix = mixture_of_two.MixtureOfTwo(PROPORTION, component1, component2)
    self.assertEqual(mix.event_shape, ())
    self.assertEqual(mix.event_shape, ())


if __name__ == '__main__':
  absltest.main()
