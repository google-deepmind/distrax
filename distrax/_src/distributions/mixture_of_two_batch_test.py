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
"""Regression tests for independent batch selectors in MixtureOfTwo."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import deterministic
from distrax._src.distributions import independent
from distrax._src.distributions import mixture_of_two
import jax
import jax.numpy as jnp
import numpy as np


class MixtureOfTwoBatchTest(parameterized.TestCase):

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('scalar_events', (2,), (), (4096,)),
      ('matrix_batch', (2, 3), (), (64, 64)),
      ('vector_events', (2,), (3,), (4096,)),
      ('matrix_events', (2, 3), (2, 3), (64, 64)),
  )
  def test_batch_members_choose_components_independently(
      self, batch_shape, event_shape, sample_shape
  ):
    component_a = independent.Independent(
        deterministic.Deterministic(jnp.zeros(batch_shape + event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )
    component_b = independent.Independent(
        deterministic.Deterministic(jnp.ones(batch_shape + event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )
    mix = mixture_of_two.MixtureOfTwo(0.5, component_a, component_b)

    samples = self.variant(mix.sample, static_argnames='sample_shape')(
        seed=jax.random.PRNGKey(42), sample_shape=sample_shape
    )
    self.assertEqual(samples.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(samples.dtype, component_a.dtype)
    samples = np.asarray(samples).reshape((4096, np.prod(batch_shape), -1))
    # Every coordinate of an event must come from the same component.
    np.testing.assert_array_equal(
        samples, np.broadcast_to(samples[..., :1], samples.shape)
    )
    labels = samples[..., 0]
    np.testing.assert_allclose(labels.mean(axis=0), 0.5, atol=0.04)
    # Independent Bernoulli(0.5) selectors disagree with probability 0.5.
    disagreement = labels[:, 1:] != labels[:, :1]
    np.testing.assert_allclose(disagreement.mean(axis=0), 0.5, atol=0.04)

  @chex.all_variants(with_pmap=False)
  def test_independent_mixture_samples_match_product_probabilities(self):
    mix = mixture_of_two.MixtureOfTwo(
        0.5,
        deterministic.Deterministic(jnp.zeros(2)),
        deterministic.Deterministic(jnp.ones(2)),
    )
    joint = independent.Independent(mix, reinterpreted_batch_ndims=1)
    samples, log_prob = self.variant(
        joint.sample_and_log_prob, static_argnames='sample_shape'
    )(seed=jax.random.PRNGKey(7), sample_shape=4096)

    outcomes, counts = np.unique(samples, axis=0, return_counts=True)
    np.testing.assert_array_equal(
        outcomes, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    np.testing.assert_allclose(counts / 4096, 0.25, atol=0.04)
    np.testing.assert_allclose(log_prob, -2 * np.log(2), rtol=1e-6)

  @chex.all_variants(with_pmap=False)
  @parameterized.product(prob_a=[0.0, 1.0], batch_shape=[(), (2, 3)])
  def test_unbatched_and_endpoint_sampling(self, prob_a, batch_shape):
    mix = mixture_of_two.MixtureOfTwo(
        prob_a,
        deterministic.Deterministic(jnp.zeros(batch_shape)),
        deterministic.Deterministic(jnp.ones(batch_shape)),
    )
    samples = self.variant(mix.sample, static_argnames='sample_shape')(
        seed=jax.random.PRNGKey(0), sample_shape=3
    )
    np.testing.assert_array_equal(
        samples, np.full((3,) + batch_shape, 1 - prob_a)
    )

  @chex.all_variants(with_pmap=False)
  def test_empty_batched_vector_samples(self):
    mix = mixture_of_two.MixtureOfTwo(
        0.5,
        independent.Independent(
            deterministic.Deterministic(jnp.zeros((2, 3))),
            reinterpreted_batch_ndims=1,
        ),
        independent.Independent(
            deterministic.Deterministic(jnp.ones((2, 3))),
            reinterpreted_batch_ndims=1,
        ),
    )
    samples = self.variant(mix.sample, static_argnames='sample_shape')(
        seed=jax.random.PRNGKey(0), sample_shape=(2, 0)
    )
    self.assertEqual(samples.shape, (2, 0, 2, 3))


if __name__ == '__main__':
  absltest.main()
