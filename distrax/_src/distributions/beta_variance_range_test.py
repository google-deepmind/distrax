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
"""Beta variances should remain representable across concentration scales."""

import decimal
from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


def exact_variance(alpha, beta):
  with decimal.localcontext() as context:
    context.prec = 80
    a = decimal.Decimal(float(alpha))
    b = decimal.Decimal(float(beta))
    return float(a * b / ((a + b) ** 2 * (a + b + 1)))


class BetaVarianceRangeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('large_symmetric', 1e20, 1e20, jnp.float32),
      ('large_asymmetric', 1e20, 1e15, jnp.float32),
      ('large_asymmetric_reversed', 1e15, 1e20, jnp.float32),
      ('tiny_symmetric', 1e-30, 1e-30, jnp.float32),
      ('tiny_asymmetric', 1e-25, 3e-25, jnp.float32),
      ('half_moderate', 100.0, 100.0, jnp.float16),
      ('half_large', 10000.0, 20000.0, jnp.float16),
      ('bfloat_large', 1e20, 1e20, jnp.bfloat16),
      ('ordinary', 2.0, 5.0, jnp.float32),
  )
  def test_variance_and_standard_deviation_match_high_precision_formula(
      self, a, b, dtype
  ):
    alpha, beta = jnp.array(a, dtype), jnp.array(b, dtype)
    expected = exact_variance(alpha, beta)
    for fn in (
        lambda x, y: distrax.Beta(x, y).variance(),
        jax.jit(lambda x, y: distrax.Beta(x, y).variance()),
    ):
      actual = fn(alpha, beta)
      self.assertEqual(actual.dtype, dtype)
      self.assertTrue(bool(jnp.isfinite(actual)))
      self.assertGreater(float(actual), 0.0)
      tolerance = (
          0.02
          if dtype == jnp.bfloat16
          else 0.003
          if dtype == jnp.float16
          else 3e-6
      )
      np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)
    np.testing.assert_allclose(
        distrax.Beta(alpha, beta).stddev(),
        np.sqrt(expected),
        rtol=tolerance,
        atol=0,
    )

  def test_broadcasted_parameters_match_a_two_component_dirichlet(self):
    alpha = jnp.array([[1e20], [1e-30], [2.0]], jnp.float32)
    beta = jnp.array([1e-30, 1.0, 1e20], jnp.float32)
    a, b = jnp.broadcast_arrays(alpha, beta)
    beta_variance = distrax.Beta(alpha, beta).variance()
    dirichlet_variance = distrax.Dirichlet(
        jnp.stack([a, b], axis=-1)
    ).covariance()[..., 0, 0]
    expected = np.array(
        [
            [exact_variance(x, y) for x, y in zip(arow, brow)]
            for arow, brow in zip(np.asarray(a), np.asarray(b))
        ]
    )
    np.testing.assert_allclose(beta_variance, expected, rtol=4e-6, atol=1e-37)
    self.assertEqual(beta_variance.shape, (3, 3))
    # The ordinary, non-boundary case agrees with the Dirichlet marginal.
    np.testing.assert_allclose(
        beta_variance[2, 1], dirichlet_variance[2, 1], rtol=3e-6
    )

  def test_moderate_concentration_values_and_gradients_are_unchanged(self):
    def original(a, b):
      total = a + b
      return a * b / (total**2 * (total + 1))

    for a, b in [(1.0, 1.0), (0.2, 0.7), (5.0, 20.0)]:
      args = (jnp.array(a), jnp.array(b))
      value, gradients = jax.jit(
          jax.value_and_grad(lambda x, y: distrax.Beta(x, y).variance(), (0, 1))
      )(*args)
      expected_value, expected_gradients = jax.value_and_grad(original, (0, 1))(
          *args
      )
      np.testing.assert_allclose(value, stats.beta.var(a, b), rtol=3e-6)
      np.testing.assert_allclose(value, expected_value, rtol=3e-6)
      np.testing.assert_allclose(
          gradients, expected_gradients, rtol=4e-6, atol=1e-8
      )

  def test_explicit_float64_range_is_preserved(self):
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
      for value in [1e-200, 1e200]:
        alpha = jnp.array(value, jnp.float64)
        actual = jax.jit(lambda x: distrax.Beta(x, x).variance())(alpha)
        self.assertEqual(actual.dtype, jnp.float64)
        np.testing.assert_allclose(
            actual, exact_variance(alpha, alpha), rtol=2e-14, atol=0
        )
    finally:
      jax.config.update('jax_enable_x64', previous)


if __name__ == '__main__':
  absltest.main()
