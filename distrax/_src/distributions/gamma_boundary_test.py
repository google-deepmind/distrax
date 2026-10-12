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
"""Gamma densities at zero agree with their exponential special case."""

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from distrax._src.utils import compat


class GammaBoundaryTest(parameterized.TestCase):

  @parameterized.product(
      rate=[0.25, 1.0, 2.0, 10.0], dtype=[jnp.float32, jnp.float64]
  )
  def test_unit_concentration_matches_the_exponential_density(
      self, rate, dtype
  ):
    with compat.enable_x64(dtype == jnp.float64):
      values = jnp.array([0.0, 1e-5, 0.1, 2.0], dtype=dtype)
      distribution = distrax.Gamma(
          jnp.array(1.0, dtype), jnp.array(rate, dtype)
      )
      expected = np.log(rate) - rate * np.asarray(values)
      for compute in (distribution.log_prob, jax.jit(distribution.log_prob)):
        actual = compute(values)
        self.assertEqual(actual.dtype, dtype)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
      np.testing.assert_allclose(
          distribution.prob(values), np.exp(expected), rtol=2e-6, atol=1e-7
      )

  def test_batched_boundary_values_match_scipy_limits(self):
    concentration = jnp.array([[0.5], [1.0], [2.0]])
    rate = jnp.array([0.25, 2.0, 10.0])
    distribution = distrax.Gamma(concentration, rate)
    actual = jax.jit(distribution.log_prob)(jnp.array(0.0))
    expected = stats.gamma.logpdf(
        0.0, np.asarray(concentration), scale=1 / np.asarray(rate)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    self.assertEqual(actual.shape, (3, 3))

  def test_rate_gradient_at_zero_is_finite(self):
    for gradient in (
        jax.grad(lambda rate: distrax.Gamma(1.0, rate).log_prob(0.0)),
        jax.jit(jax.grad(lambda rate: distrax.Gamma(1.0, rate).log_prob(0.0))),
    ):
      np.testing.assert_allclose(gradient(jnp.array(2.0)), 0.5, rtol=1e-6)

  def test_independent_exponentials_have_finite_joint_log_density_at_origin(
      self,
  ):
    rate = jnp.array([0.5, 2.0, 3.0])
    distribution = distrax.Independent(distrax.Gamma(jnp.ones(3), rate), 1)
    expected = np.log(np.asarray(rate)).sum()
    np.testing.assert_allclose(
        jax.jit(distribution.log_prob)(jnp.zeros(3)), expected, atol=2e-6
    )

  @parameterized.parameters(0.5, 1.0, 2.0)
  def test_positive_values_and_derivatives_are_unchanged(self, concentration):
    rate = 2.0
    value = jnp.array(0.75)
    distribution = distrax.Gamma(concentration, rate)
    actual, gradient = jax.value_and_grad(distribution.log_prob)(value)
    expected = stats.gamma.logpdf(float(value), concentration, scale=1 / rate)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(
        gradient, (concentration - 1) / float(value) - rate, rtol=1e-6
    )


if __name__ == '__main__':
  absltest.main()
