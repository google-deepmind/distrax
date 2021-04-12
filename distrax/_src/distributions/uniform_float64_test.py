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
"""Tests for `uniform.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import uniform
import jax
from jax.config import config as jax_config
import jax.numpy as jnp


def setUpModule():
  jax_config.update('jax_enable_x64', True)


class UniformFloat64Test(chex.TestCase):

  def _assert_dtypes(self, dist, dtype):
    """Asserts dist methods' outputs' datatypes."""
    # Sanity check to make sure float64 is enabled.
    x_64 = jnp.zeros([])
    self.assertEqual(jnp.float64, x_64.dtype)

    key = jax.random.PRNGKey(1729)
    z, log_prob = self.variant(
        lambda: dist.sample_and_log_prob(seed=key, sample_shape=[3]))()
    z2 = self.variant(
        lambda: dist.sample(seed=key, sample_shape=[3]))()
    self.assertEqual(dtype, z.dtype)
    self.assertEqual(dtype, z2.dtype)
    self.assertEqual(dtype, log_prob.dtype)
    self.assertEqual(dtype, self.variant(dist.log_prob)(z).dtype)
    self.assertEqual(dtype, self.variant(dist.prob)(z).dtype)
    self.assertEqual(dtype, self.variant(dist.log_cdf)(z).dtype)
    self.assertEqual(dtype, self.variant(dist.cdf)(z).dtype)
    self.assertEqual(dtype, self.variant(dist.entropy)().dtype)
    self.assertEqual(dtype, self.variant(dist.mean)().dtype)
    self.assertEqual(dtype, self.variant(dist.median)().dtype)
    self.assertEqual(dtype, self.variant(dist.stddev)().dtype)
    self.assertEqual(dtype, self.variant(dist.variance)().dtype)
    self.assertEqual(dtype, dist.low.dtype)
    self.assertEqual(dtype, dist.high.dtype)
    self.assertEqual(dtype, dist.range.dtype)
    self.assertEqual(dtype, dist.dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_dtype(self, dtype):
    dist = uniform.Uniform(low=jnp.zeros([], dtype), high=jnp.ones([], dtype))
    self._assert_dtypes(dist, dtype)


if __name__ == '__main__':
  absltest.main()
