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
"""Regression tests for multinomial entropy with large trial counts."""

from absl.testing import absltest
from absl.testing import parameterized
from distrax._src.distributions import multinomial
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


class MultinomialEntropyTest(parameterized.TestCase):

  @parameterized.product(
      total_count=(150, 200, 512),
      probs=((0.2, 0.3, 0.5), (0.0, 0.25, 0.75), (1.0, 0.0, 0.0)),
      implementation=('public', 'vectorized'),
      compiled=(False, True),
  )
  def test_large_count_entropy(
      self, total_count, probs, implementation, compiled
  ):
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    dist = multinomial.Multinomial(total_count, probs=jnp.asarray(probs, dtype))
    if implementation == 'public':
      entropy = dist.entropy
    else:
      entropy = lambda: dist._entropy_scalar(
          total_count, dist.probs, dist.log_of_probs
      )
    actual = jax.jit(entropy)() if compiled else entropy()
    expected = stats.multinomial(total_count, np.asarray(probs)).entropy()
    # The result subtracts O(n log n) terms in the output dtype.
    # At these counts float32 roundoff is bounded at the 1e-3 scale.
    tolerance = 1e-8 if jax.config.x64_enabled else 1e-3
    self.assertTrue(np.isfinite(actual))
    np.testing.assert_allclose(
        actual,
        expected,
        atol=tolerance,
        rtol=1e-9 if jax.config.x64_enabled else 1e-4,
    )
    self.assertEqual(actual.dtype, jnp.dtype(dtype))

  @parameterized.parameters(0, 1, 10)
  def test_small_counts_match_reference(self, total_count):
    probs = np.array([0.2, 0.3, 0.5])
    dist = multinomial.Multinomial(total_count, probs=probs)
    expected = stats.multinomial(total_count, probs).entropy()
    np.testing.assert_allclose(dist.entropy(), expected, atol=2e-5, rtol=1e-4)

  def test_jit_broadcasts_counts_and_preserves_batch_shape(self):
    counts = jnp.array([[150], [200]])
    probs = jnp.array([[0.2, 0.3, 0.5], [0.0, 0.25, 0.75]])
    actual = jax.jit(
        lambda n, p: multinomial.Multinomial(n, probs=p).entropy()
    )(counts, probs)
    expected = np.array(
        [
            [stats.multinomial(n, p).entropy() for p in np.asarray(probs)]
            for n in (150, 200)
        ]
    )
    self.assertEqual(actual.shape, (2, 2))
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
