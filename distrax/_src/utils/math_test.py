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
"""Tests for `math.py`."""

from absl.testing import absltest

from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special


class MathTest(absltest.TestCase):

  def test_multiply_no_nan(self):
    zero = jnp.zeros(())
    nan = zero / zero
    self.assertTrue(jnp.isnan(math.multiply_no_nan(zero, nan)))
    self.assertFalse(jnp.isnan(math.multiply_no_nan(nan, zero)))

  def test_multiply_no_nan_grads(self):
    x = -jnp.inf
    y = 0.
    self.assertEqual(math.multiply_no_nan(x, y), 0.)
    grad_fn = jax.grad(
        lambda inputs: math.multiply_no_nan(inputs[0], inputs[1]))
    np.testing.assert_allclose(grad_fn((x, y)), (y, x), rtol=1e-3)

  def test_power_no_nan(self):
    zero = jnp.zeros(())
    nan = zero / zero
    self.assertTrue(jnp.isnan(math.power_no_nan(zero, nan)))
    self.assertFalse(jnp.isnan(math.power_no_nan(nan, zero)))

  def test_power_no_nan_grads(self):
    x = np.exp(1.)
    y = 0.
    self.assertEqual(math.power_no_nan(x, y), 1.)
    grad_fn = jax.grad(
        lambda inputs: math.power_no_nan(inputs[0], inputs[1]))
    np.testing.assert_allclose(grad_fn((x, y)), (0., 1.), rtol=1e-3)

  def test_normalize_probs(self):
    pre_normalised_probs = jnp.array([0.4, 0.4, 0., 0.2])
    unnormalised_probs = jnp.array([4., 4., 0., 2.])
    expected_probs = jnp.array([0.4, 0.4, 0., 0.2])
    np.testing.assert_array_almost_equal(
        math.normalize(probs=pre_normalised_probs), expected_probs)
    np.testing.assert_array_almost_equal(
        math.normalize(probs=unnormalised_probs), expected_probs)

  def test_normalize_logits(self):
    unnormalised_logits = jnp.array([1., -1., 3.])
    expected_logits = jax.nn.log_softmax(unnormalised_logits, axis=-1)
    np.testing.assert_array_almost_equal(
        math.normalize(logits=unnormalised_logits), expected_logits)
    np.testing.assert_array_almost_equal(
        math.normalize(logits=expected_logits), expected_logits)

  def test_sum_last(self):
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 3, 4))
    np.testing.assert_array_equal(math.sum_last(x, 0), x)
    np.testing.assert_array_equal(math.sum_last(x, 1), x.sum(-1))
    np.testing.assert_array_equal(math.sum_last(x, 2), x.sum((-2, -1)))
    np.testing.assert_array_equal(math.sum_last(x, 3), x.sum())

  def test_log_expbig_minus_expsmall(self):
    small = jax.random.normal(jax.random.PRNGKey(42), (2, 3, 4))
    big = small + jax.random.uniform(jax.random.PRNGKey(43), (2, 3, 4))
    expected_result = np.log(np.exp(big) - np.exp(small))
    np.testing.assert_allclose(
        math.log_expbig_minus_expsmall(big, small), expected_result, atol=1e-4)

  def test_log_beta(self):
    a = jnp.abs(jax.random.normal(jax.random.PRNGKey(42), (2, 3, 4)))
    b = jnp.abs(jax.random.normal(jax.random.PRNGKey(43), (3, 4)))
    expected_result = scipy.special.betaln(a, b)
    np.testing.assert_allclose(math.log_beta(a, b), expected_result, atol=2e-4)

  def test_log_beta_bivariate(self):
    a = jnp.abs(jax.random.normal(jax.random.PRNGKey(42), (4, 3, 2)))
    expected_result = scipy.special.betaln(a[..., 0], a[..., 1])
    np.testing.assert_allclose(
        math.log_beta_multivariate(a), expected_result, atol=2e-4)

  def test_log_beta_multivariate(self):
    a = jnp.abs(jax.random.normal(jax.random.PRNGKey(42), (2, 3, 4)))
    expected_result = (jnp.sum(scipy.special.gammaln(a), axis=-1)
                       - scipy.special.gammaln(jnp.sum(a, axis=-1)))
    np.testing.assert_allclose(
        math.log_beta_multivariate(a), expected_result, atol=1e-3)

if __name__ == '__main__':
  absltest.main()
