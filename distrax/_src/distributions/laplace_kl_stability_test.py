# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Numerical regressions for Laplace KL divergence."""

from decimal import Decimal
from decimal import localcontext

from absl.testing import absltest
from absl.testing import parameterized
from distrax._src.distributions import laplace
from distrax._src.utils import compat
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
_MODES = ('distrax_to_distrax', 'distrax_to_tfp', 'tfp_to_distrax')


def _distributions(mode, loc1, scale1, loc2, scale2):
  first = tfd.Laplace if mode == 'tfp_to_distrax' else laplace.Laplace
  second = tfd.Laplace if mode == 'distrax_to_tfp' else laplace.Laplace
  return first(loc1, scale1), second(loc2, scale2)


def _reference(loc1, scale1, loc2, scale2):
  """Evaluates the original dimensional formula and derivatives in Decimal."""
  with localcontext() as context:
    context.prec = 80
    m1, a, m2, b = (
        Decimal.from_float(float(x)) for x in (loc1, scale1, loc2, scale2)
    )
    distance = abs(m1 - m2)
    decay = (-distance / a).exp()
    kl = (b / a).ln() + distance / b - 1 + a / b * decay
    entropy = 1 + (2 * a).ln()
    sign = Decimal((m1 > m2) - (m1 < m2))
    loc_grad = sign * (1 - decay) / b
    gradient = [
        loc_grad,
        -1 / a + decay / b * (1 + distance / a),
        -loc_grad,
        (b - distance - a * decay) / b**2,
    ]
    return float(kl), float(kl + entropy), np.array(gradient, np.float64)


class LaplaceKlStabilityTest(parameterized.TestCase):

  @parameterized.product(
      mode=_MODES, dtype=(jnp.float32, jnp.float64), compiled=(False, True)
  )
  def test_nearby_locations(self, mode, dtype, compiled):
    with compat.enable_x64(dtype == jnp.float64):
      small, medium = (1e-4, 1e-3) if dtype == jnp.float32 else (1e-8, 1e-7)
      loc1 = jnp.array([[0.0], [0.5]], dtype)
      loc2 = jnp.array(
          [[0.0, small, -small, medium, -medium, 0.25, -0.25]], dtype
      )
      scale = jnp.asarray(1.0, dtype)

      def divergence(first, second):
        a, b = _distributions(mode, first, scale, second, scale)
        return a.kl_divergence(b)

      if compiled:
        divergence = jax.jit(divergence)
      actual = divergence(loc1, loc2)
      expected = np.array(
          [
              [_reference(x, 1.0, y, 1.0)[0] for y in np.asarray(loc2)[0]]
              for x in np.asarray(loc1)[:, 0]
          ]
      )
      self.assertEqual(actual.shape, (2, 7))
      self.assertEqual(actual.dtype, dtype)
      self.assertTrue(np.all(np.isfinite(actual)))
      self.assertTrue(np.all(np.asarray(actual) >= 0))
      tolerance = 2e-3 if dtype == jnp.float32 else 2e-7
      np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)

  @parameterized.product(
      mode=_MODES, dtype=(jnp.float32, jnp.float64), compiled=(False, True)
  )
  def test_values_and_parameter_gradients(self, mode, dtype, compiled):
    with compat.enable_x64(dtype == jnp.float64):
      parameters = jnp.array([0.2, 0.8, -0.1, 1.3], dtype)

      def objectives(p):
        a, b = _distributions(mode, *p)
        return jnp.stack([a.kl_divergence(b), a.cross_entropy(b)])

      def evaluate(p):
        return objectives(p), jax.jacrev(objectives)(p)

      if compiled:
        evaluate = jax.jit(evaluate)
      values, gradients = evaluate(parameters)
      kl, cross_entropy, kl_gradient = _reference(*np.asarray(parameters))
      ce_gradient = kl_gradient.copy()
      ce_gradient[1] += 1.0 / float(parameters[1])
      self.assertEqual(values.dtype, dtype)
      self.assertEqual(gradients.dtype, dtype)
      self.assertTrue(np.all(np.isfinite(values)))
      self.assertTrue(np.all(np.isfinite(gradients)))
      tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
      absolute = 2e-6 if dtype == jnp.float32 else 1e-12
      np.testing.assert_allclose(
          values, [kl, cross_entropy], rtol=tolerance, atol=absolute
      )
      np.testing.assert_allclose(
          gradients, [kl_gradient, ce_gradient], rtol=tolerance, atol=absolute
      )


if __name__ == '__main__':
  absltest.main()
