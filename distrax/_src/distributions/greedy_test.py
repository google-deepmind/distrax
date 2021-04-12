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
"""Tests for `greedy.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import greedy
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np

RTOL = 2e-3


class GreedyTest(equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(greedy.Greedy)
    self.preferences = jnp.array([0., 4., -1., 4.])
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  def test_parameters_from_preferences(self):
    dist = self.distrax_cls(preferences=self.preferences)
    expected_probs = jnp.array([0., 0.5, 0., 0.5])
    self.assertion_fn(dist.logits, jnp.log(expected_probs))
    self.assertion_fn(dist.probs, expected_probs)

  def test_num_categories(self):
    dist = self.distrax_cls(preferences=self.preferences)
    np.testing.assert_equal(dist.num_categories, len(self.preferences))

  @chex.all_variants
  @parameterized.named_parameters(
      ('int32', jnp.int32),
      ('int64', jnp.int64),
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(preferences=self.preferences, dtype=dtype)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  def test_jittable(self):
    super()._test_jittable((np.array([0., 4., -1., 4.]),))


if __name__ == '__main__':
  absltest.main()
