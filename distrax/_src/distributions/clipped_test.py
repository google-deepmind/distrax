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
"""Tests for `clipped.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import clipped
from distrax._src.distributions import logistic
from distrax._src.distributions import normal
import jax
import jax.numpy as jnp
import numpy as np


MINIMUM = -1.0
MAXIMUM = 1.0
LOC = MINIMUM
SCALE = 0.1
SIZE = 3


class ClippedTest(parameterized.TestCase):

  @parameterized.parameters([
      [clipped.ClippedLogistic, logistic.Logistic],
      [clipped.ClippedNormal, normal.Normal],
  ])
  def test_clipped_logprob(self, factory, unclipped_factory):
    distribution = factory(
        loc=LOC, scale=SCALE, minimum=MINIMUM, maximum=MAXIMUM)
    unclipped = unclipped_factory(loc=LOC, scale=SCALE)

    np.testing.assert_allclose(
        unclipped.log_prob(0.0),
        distribution.log_prob(0.0))
    np.testing.assert_allclose(
        unclipped.log_prob(0.8),
        distribution.log_prob(0.8))

    # Testing outside of the boundary.
    self.assertEqual(-np.inf, distribution.log_prob(MINIMUM - 0.1))
    self.assertEqual(-np.inf, distribution.log_prob(MAXIMUM + 0.1))

  @parameterized.parameters([
      [clipped.ClippedLogistic, logistic.Logistic],
      [clipped.ClippedNormal, normal.Normal],
  ])
  def test_batched_clipped_logprob(self, factory, unclipped_factory):
    distribution = factory(
        loc=jnp.array([LOC]*SIZE),
        scale=jnp.array([SCALE]*SIZE),
        minimum=MINIMUM,
        maximum=MAXIMUM)
    unclipped = unclipped_factory(loc=LOC, scale=SCALE)

    np.testing.assert_allclose(
        unclipped.log_prob(jnp.array([0.0]*SIZE)),
        distribution.log_prob(jnp.array([0.0]*SIZE)))
    np.testing.assert_allclose(
        unclipped.log_prob(jnp.array([0.8]*SIZE)),
        distribution.log_prob(jnp.array([0.8]*SIZE)))

    # Testing outside of the boundary.
    np.testing.assert_allclose(
        -np.inf, distribution.log_prob(jnp.array([MINIMUM - 0.1]*SIZE)))
    np.testing.assert_allclose(
        -np.inf, distribution.log_prob(jnp.array([MAXIMUM + 0.1]*SIZE)))

  @parameterized.parameters([
      [clipped.ClippedLogistic, logistic.Logistic],
      [clipped.ClippedNormal, normal.Normal],
  ])
  def test_clipped_sampled_and_logprob(self, factory, unclipped_factory):
    distribution = factory(
        loc=LOC, scale=SCALE, minimum=MINIMUM, maximum=MAXIMUM)
    unclipped = unclipped_factory(loc=LOC, scale=SCALE)

    for rng in jax.random.split(jax.random.PRNGKey(42), 5):
      sample, log_prob = distribution.sample_and_log_prob(seed=rng)
      unclipped_sample, unclipped_log_prob = unclipped.sample_and_log_prob(
          seed=rng)
      if float(unclipped_sample) > MAXIMUM:
        np.testing.assert_allclose(sample, MAXIMUM, atol=1e-5)
      elif float(unclipped_sample) < MINIMUM:
        np.testing.assert_allclose(sample, MINIMUM, atol=1e-5)
      else:
        np.testing.assert_allclose(sample, unclipped_sample, atol=1e-5)
        np.testing.assert_allclose(log_prob, unclipped_log_prob, atol=1e-5)

  @parameterized.parameters([
      [clipped.ClippedLogistic, logistic.Logistic],
      [clipped.ClippedNormal, normal.Normal],
  ])
  def test_clipped_sample(self, factory, unclipped_factory):
    distribution = factory(
        loc=LOC, scale=SCALE, minimum=MINIMUM, maximum=MAXIMUM)
    unclipped = unclipped_factory(loc=LOC, scale=SCALE)

    for rng in jax.random.split(jax.random.PRNGKey(42), 5):
      sample = distribution.sample(seed=rng)
      unclipped_sample = unclipped.sample(seed=rng)
      if float(unclipped_sample) > MAXIMUM:
        np.testing.assert_allclose(sample, MAXIMUM, atol=1e-5)
      elif float(unclipped_sample) < MINIMUM:
        np.testing.assert_allclose(sample, MINIMUM, atol=1e-5)
      else:
        np.testing.assert_allclose(sample, unclipped_sample, atol=1e-5)

  @parameterized.parameters([
      [clipped.ClippedLogistic],
      [clipped.ClippedNormal],
  ])
  def test_extremes(self, factory):
    minimum = -1.0
    maximum = 1.0
    scale = 0.01

    # Using extreme loc.
    distribution = factory(
        loc=minimum, scale=scale, minimum=minimum, maximum=maximum)
    self.assertTrue(np.isfinite(distribution.log_prob(minimum)))
    self.assertTrue(np.isfinite(distribution.log_prob(maximum)))

    distribution = factory(
        loc=maximum, scale=scale, minimum=minimum, maximum=maximum)
    self.assertTrue(np.isfinite(distribution.log_prob(minimum)))
    self.assertTrue(np.isfinite(distribution.log_prob(maximum)))

  def test_jitable(self):
    minimum = -1.0
    maximum = 1.0
    loc = minimum
    scale = 0.1

    @jax.jit
    def jitted_function(event, dist):
      return dist.log_prob(event)

    dist = clipped.ClippedLogistic(
        loc=loc, scale=scale, minimum=minimum, maximum=maximum)
    event = dist.sample(seed=jax.random.PRNGKey(4242))
    log_prob = dist.log_prob(event)
    jitted_log_prob = jitted_function(event, dist)

    chex.assert_trees_all_close(
        jitted_log_prob, log_prob, atol=1e-4, rtol=1e-4)

  def test_properties(self):
    dist = clipped.ClippedLogistic(
        loc=LOC, scale=SCALE, minimum=MINIMUM, maximum=MAXIMUM)
    np.testing.assert_allclose(dist.minimum, MINIMUM, atol=1e-5)
    np.testing.assert_allclose(dist.maximum, MAXIMUM, atol=1e-5)
    dist = clipped.ClippedLogistic(
        loc=jnp.array([LOC]*SIZE),
        scale=jnp.array([SCALE]*SIZE),
        minimum=MINIMUM,
        maximum=MAXIMUM)
    np.testing.assert_allclose(dist.minimum, MINIMUM, atol=1e-5)
    np.testing.assert_allclose(dist.maximum, MAXIMUM, atol=1e-5)

  def test_min_max_broadcasting(self):
    dist = clipped.ClippedLogistic(
        loc=jnp.array([LOC]*SIZE),
        scale=jnp.array([SCALE]*SIZE),
        minimum=MINIMUM,
        maximum=MAXIMUM)
    self.assertEqual(dist.minimum.shape, (SIZE,))
    self.assertEqual(dist.minimum.shape, (SIZE,))

  def test_batch_shape(self):
    dist = clipped.ClippedLogistic(
        loc=jnp.array([LOC]*SIZE),
        scale=jnp.array([SCALE]*SIZE),
        minimum=MINIMUM,
        maximum=MAXIMUM)
    self.assertEqual(dist.batch_shape, (SIZE,))
    self.assertEqual(dist.batch_shape, (SIZE,))

  def test_event_shape(self):
    dist = clipped.ClippedLogistic(
        loc=jnp.array([LOC]*SIZE),
        scale=jnp.array([SCALE]*SIZE),
        minimum=jnp.array([MINIMUM]*SIZE),
        maximum=jnp.array([MAXIMUM]*SIZE))
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.event_shape, ())


if __name__ == '__main__':
  absltest.main()
