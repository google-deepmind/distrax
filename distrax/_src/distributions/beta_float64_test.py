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
"""Float64 regression tests for Beta log probabilities."""

from absl.testing import absltest
from distrax._src.distributions import beta
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np


def setUpModule():
  jax.config.update('jax_enable_x64', True)


class BetaFloat64Test(absltest.TestCase):

  def test_log_prob_unbalanced_parameters(self):
    concentration = jnp.array([1e16, 1e17], dtype=jnp.float64)
    value = jnp.array([[1e-16], [1e-17]], dtype=jnp.float64)
    b, x = np.asarray(concentration), np.asarray(value)
    expected = np.log(b) + (b - 1.) * np.log1p(-x)
    log_prob = lambda b, x: beta.Beta(jnp.float64(1.), b).log_prob(x)
    for fn in [log_prob, jax.jit(log_prob)]:
      actual = fn(concentration, value)
      self.assertEqual(actual.dtype, jnp.float64)
      np.testing.assert_allclose(actual, expected, rtol=1e-12)
    for fn in [math.log_beta, jax.jit(math.log_beta)]:
      np.testing.assert_allclose(fn(jnp.float64(1.), concentration),
                                 -np.log(b), rtol=1e-12)


if __name__ == '__main__':
  absltest.main()
