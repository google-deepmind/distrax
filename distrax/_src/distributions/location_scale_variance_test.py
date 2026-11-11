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
"""Numerical range tests for Logistic and Gumbel variances."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions import gumbel
from distrax._src.distributions import independent
from distrax._src.distributions import logistic
from distrax._src.utils import compat
import jax
import jax.numpy as jnp
import numpy as np


class LocationScaleVarianceTest(chex.TestCase):

  @chex.all_variants(with_pmap=False)
  @parameterized.product(
      distribution=[(logistic.Logistic, 3.0), (gumbel.Gumbel, 6.0)],
      dtype=[jnp.float16, jnp.float32, jnp.float64],
      batched=[False, True],
  )
  def test_variance_representable_range(self, distribution, dtype, batched):
    cls, denominator = distribution
    coefficient = np.pi**2 / denominator
    finfo = np.finfo(dtype)
    boundary = np.sqrt(float(finfo.max) / coefficient)
    with compat.enable_x64(dtype == jnp.float64):
      if batched:
        loc = jnp.zeros((2, 1), dtype)
        scale = jnp.asarray(
            [
                0.5,
                2.0 * np.sqrt(float(finfo.tiny) / coefficient),
                0.9 * boundary,
                1.1 * boundary,
            ],
            dtype,
        )

        def variance(loc, scale):
          return independent.Independent(cls(loc, scale), 1).variance()

        expected_shape = (2, 4)
      else:
        loc = jnp.zeros((), dtype)
        scale = jnp.asarray(0.9 * boundary, dtype)
        variance = lambda loc, scale: cls(loc, scale).variance()
        expected_shape = ()
      actual = self.variant(variance)(loc, scale)
      with np.errstate(over='ignore'):
        expected = np.asarray(scale, np.float64) ** 2 * coefficient
        expected = np.asarray(expected, np.dtype(dtype))
      expected = np.broadcast_to(expected, expected_shape)
      self.assertEqual(actual.shape, expected_shape)
      self.assertEqual(actual.dtype, dtype)
      np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
      rtol = {jnp.float16: 3e-3, jnp.float32: 1e-6, jnp.float64: 1e-12}[dtype]
      np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0)

  @chex.all_variants(with_pmap=False)
  @parameterized.product(
      distribution=[(logistic.Logistic, 3.0), (gumbel.Gumbel, 6.0)],
      dtype=[jnp.float32, jnp.float64],
  )
  def test_variance_analytic_gradients(self, distribution, dtype):
    cls, denominator = distribution
    coefficient = np.pi**2 / denominator
    boundary = np.sqrt(float(np.finfo(dtype).max) / coefficient)
    with compat.enable_x64(dtype == jnp.float64):
      loc = jnp.zeros((), dtype)
      scale = jnp.asarray([0.01, 1.0, 0.5 * boundary], dtype)
      gradient = jax.grad(
          lambda loc, scale: jnp.sum(cls(loc, scale).variance()), argnums=(0, 1)
      )
      loc_gradient, scale_gradient = self.variant(gradient)(loc, scale)
      expected = 2.0 * coefficient * np.asarray(scale, np.float64)
      np.testing.assert_array_equal(loc_gradient, 0.0)
      self.assertEqual(scale_gradient.shape, scale.shape)
      self.assertEqual(scale_gradient.dtype, dtype)
      np.testing.assert_allclose(
          scale_gradient,
          expected,
          rtol=1e-6 if dtype == jnp.float32 else 1e-12,
          atol=0.0,
      )


if __name__ == '__main__':
  absltest.main()
