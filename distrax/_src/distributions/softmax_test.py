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
"""Tests for `softmax.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import softmax
from distrax._src.utils import equivalence
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np


RTOL = 2e-3


class SoftmaxUnitTemperatureTest(
    equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(softmax.Softmax)
    self.temperature = 1.
    self.probs = jnp.array([0.2, 0.4, 0.1, 0.3])
    self.logits = jnp.log(self.probs)
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  def test_num_categories(self):
    dist = self.distrax_cls(logits=self.logits)
    np.testing.assert_equal(dist.num_categories, len(self.logits))

  def test_parameters(self):
    dist = self.distrax_cls(logits=self.logits)
    self.assertion_fn(dist.logits, self.logits)
    self.assertion_fn(dist.probs, self.probs)


class SoftmaxTest(equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(softmax.Softmax)
    self.temperature = 10.
    self.logits = jnp.array([2., 4., 1., 3.])
    self.probs = jax.nn.softmax(self.logits / self.temperature)
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  def test_num_categories(self):
    dist = self.distrax_cls(logits=self.logits, temperature=self.temperature)
    np.testing.assert_equal(dist.num_categories, len(self.logits))

  def test_parameters(self):
    dist = self.distrax_cls(logits=self.logits, temperature=self.temperature)
    self.assertion_fn(
        dist.logits, math.normalize(logits=self.logits / self.temperature))
    self.assertion_fn(dist.probs, self.probs)

  @chex.all_variants
  @parameterized.named_parameters(
      ('int32', jnp.int32),
      ('int64', jnp.int64),
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(
        logits=self.logits, temperature=self.temperature, dtype=dtype)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  def test_jittable(self):
    super()._test_jittable((np.array([2., 4., 1., 3.]),))


if __name__ == '__main__':
  absltest.main()
