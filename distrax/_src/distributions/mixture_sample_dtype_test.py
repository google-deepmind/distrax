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
"""Mixture sampling preserves the support and dtype of its components."""

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np


class MixtureSampleDtypeTest(parameterized.TestCase):

  @parameterized.parameters(
      jnp.int8,
      jnp.int16,
      jnp.int32,
      jnp.uint32,
      jnp.bool_,
      jnp.float16,
      jnp.bfloat16,
      jnp.float32,
  )
  def test_sample_dtype_matches_component_dtype(self, dtype):
    locations = jnp.array([1, 0], dtype=dtype)
    mixture = distrax.MixtureSameFamily(
        distrax.Categorical(probs=jnp.array([1.0, 0.0])),
        distrax.Deterministic(locations),
    )
    for draw in (
        lambda key: mixture.sample(seed=key, sample_shape=(2, 3)),
        jax.jit(lambda key: mixture.sample(seed=key, sample_shape=(2, 3))),
    ):
      actual = draw(jax.random.key(0))
      self.assertEqual(actual.dtype, locations.dtype)
      np.testing.assert_array_equal(actual, np.ones((2, 3)))

  @parameterized.parameters(16777217, -16777217, 2147483647)
  def test_integer_values_are_not_rounded_through_float(self, value):
    locations = jnp.array([value, 0], dtype=jnp.int32)
    mixture = distrax.MixtureSameFamily(
        distrax.Categorical(probs=jnp.array([1.0, 0.0])),
        distrax.Deterministic(locations),
    )
    actual = mixture.sample(seed=jax.random.key(0), sample_shape=4)
    self.assertEqual(actual.dtype, jnp.int32)
    np.testing.assert_array_equal(actual, np.full(4, value, dtype=np.int32))

  def test_batched_vector_events_match_selected_component_exactly(self):
    locations = jnp.array(
        [
            [[16777217, -16777217], [3, 4], [5, 6]],
            [[7, 8], [2147483647, 9], [10, 11]],
        ],
        jnp.int32,
    )
    mixing = distrax.Categorical(
        probs=jnp.array([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    )
    components = distrax.Independent(distrax.Deterministic(locations), 1)
    mixture = distrax.MixtureSameFamily(mixing, components)
    key = jax.random.key(7)
    mix_key, components_key = jax.random.split(key)
    indices = mixing.sample(seed=mix_key, sample_shape=11)
    all_samples = components.sample(seed=components_key, sample_shape=11)
    expected = jnp.take_along_axis(
        all_samples, indices[..., None, None], axis=2
    ).squeeze(2)
    actual = jax.jit(lambda k: mixture.sample(seed=k, sample_shape=11))(key)
    self.assertEqual(actual.dtype, jnp.int32)
    np.testing.assert_array_equal(actual, expected)

  def test_discrete_samples_remain_valid_array_indices(self):
    components = distrax.Categorical(
        probs=jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    mixture = distrax.MixtureSameFamily(
        distrax.Categorical(probs=jnp.array([0.5, 0.5])), components
    )
    samples, log_prob = mixture.sample_and_log_prob(
        seed=jax.random.key(9), sample_shape=30
    )
    self.assertEqual(samples.dtype, components.dtype)
    table = jnp.array([2.0, 3.0, 5.0])
    selected = jax.jit(lambda x: table[x])(samples)
    np.testing.assert_array_equal(
        selected, np.where(np.asarray(samples) == 0, 2.0, 5.0)
    )
    np.testing.assert_allclose(log_prob, np.log(0.5), rtol=1e-6)

  def test_float_samples_and_reparameterized_gradients_are_unchanged(self):
    key = jax.random.key(11)
    weights = distrax.Categorical(probs=jnp.array([0.2, 0.8]))

    def actual(loc):
      mixture = distrax.MixtureSameFamily(
          weights, distrax.Normal(loc, jnp.ones(2))
      )
      return mixture.sample(seed=key, sample_shape=20).sum()

    def reference(loc):
      mix_key, sample_key = jax.random.split(key)
      indices = weights.sample(seed=mix_key, sample_shape=20)
      samples = distrax.Normal(loc, jnp.ones(2)).sample(
          seed=sample_key, sample_shape=20
      )
      return jnp.take_along_axis(samples, indices[:, None], axis=1).sum()

    loc = jnp.array([2.0, 3.0])
    for a, b in zip(
        jax.value_and_grad(actual)(loc), jax.value_and_grad(reference)(loc)
    ):
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=0)

  def test_empty_sample_shape_keeps_integer_dtype(self):
    mixture = distrax.MixtureSameFamily(
        distrax.Categorical(probs=jnp.array([1.0, 0.0])),
        distrax.Deterministic(jnp.array([3, 5], jnp.int16)),
    )
    actual = mixture.sample(seed=jax.random.key(0), sample_shape=0)
    self.assertEqual(actual.shape, (0,))
    self.assertEqual(actual.dtype, jnp.int16)


if __name__ == '__main__':
  absltest.main()
