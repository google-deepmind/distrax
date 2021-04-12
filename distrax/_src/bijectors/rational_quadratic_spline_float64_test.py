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
"""Tests for `rational_quadratic_spline.py`.

Float64 is enabled in these tests. We keep them separate from other tests to
avoid interfering with types elsewhere.
"""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import rational_quadratic_spline
from jax.config import config as jax_config
import jax.numpy as jnp


def setUpModule():
  jax_config.update('jax_enable_x64', True)


class RationalQuadraticSplineFloat64Test(chex.TestCase):
  """Tests for rational quadratic spline that use float64."""

  def _assert_dtypes(self, bij, x, dtype):
    """Asserts dtypes."""
    # Sanity check to make sure float64 is enabled.
    x_64 = jnp.zeros([])
    self.assertEqual(jnp.float64, x_64.dtype)

    y, logd = self.variant(bij.forward_and_log_det)(x)
    self.assertEqual(dtype, y.dtype)
    self.assertEqual(dtype, logd.dtype)
    y, logd = self.variant(bij.inverse_and_log_det)(x)
    self.assertEqual(dtype, y.dtype)
    self.assertEqual(dtype, logd.dtype)

  @chex.all_variants
  @parameterized.product(
      dtypes=[(jnp.float32, jnp.float32, jnp.float32),
              (jnp.float32, jnp.float64, jnp.float64),
              (jnp.float64, jnp.float32, jnp.float64),
              (jnp.float64, jnp.float64, jnp.float64)],
      boundary_slopes=['unconstrained', 'lower_identity', 'upper_identity',
                       'identity', 'circular'])
  def test_dtypes(self, dtypes, boundary_slopes):
    x_dtype, params_dtype, result_dtype = dtypes
    x = jnp.zeros([3], x_dtype)
    self.assertEqual(x_dtype, x.dtype)
    spline = rational_quadratic_spline.RationalQuadraticSpline(
        jnp.zeros([25], params_dtype), 0., 1., boundary_slopes=boundary_slopes)
    self._assert_dtypes(spline, x, result_dtype)


if __name__ == '__main__':
  absltest.main()
