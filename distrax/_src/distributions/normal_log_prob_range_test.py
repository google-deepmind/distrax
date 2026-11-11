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
"""Regression tests for representable Gaussian log probabilities."""

import contextlib
import decimal
import math

from absl.testing import absltest
from absl.testing import parameterized
from distrax._src.distributions import independent
from distrax._src.distributions import mvn_diag
from distrax._src.distributions import normal
import jax
import jax.numpy as jnp
import numpy as np

DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)


@contextlib.contextmanager
def x64_enabled(enabled):
  """Uses the public config API and restores the caller's precision setting."""
  previous = jax.config.jax_enable_x64
  jax.config.update('jax_enable_x64', enabled)
  try:
    yield
  finally:
    jax.config.update('jax_enable_x64', previous)


def reference(loc, scale, value):
  """Evaluates the density and derivatives using Decimal intermediates."""
  with decimal.localcontext() as context:
    context.prec = 80
    a, s, x = [
        decimal.Decimal.from_float(float(v)) for v in (loc, scale, value)
    ]
    z = (x - a) / s
    log_prob = (
        -z * z / 2 - s.ln() - decimal.Decimal(str(math.log(2 * math.pi) / 2))
    )
    gradients = (z / s, (z * z - 1) / s, -z / s)
    return float(log_prob), [float(g) for g in gradients]


def residual(dtype):
  if dtype == jnp.float16:
    return 300.0
  if dtype == jnp.float64:
    return 1.5e154
  return 2e19


def tolerance(dtype):
  if dtype == jnp.bfloat16:
    return 3e-2
  if dtype == jnp.float16:
    return 3e-3
  return 4e-15 if dtype == jnp.float64 else 2e-6


class NormalLogProbRangeTest(parameterized.TestCase):

  @parameterized.product(dtype=DTYPES, compiled=(False, True))
  def test_log_prob(self, dtype, compiled):
    with x64_enabled(dtype == jnp.float64):
      loc = jnp.asarray([-1.0, 0.0, 1.0], dtype)
      scale = jnp.asarray(2.0, dtype)
      values = jnp.asarray(
          [-2 * residual(dtype), 0.0, 2 * residual(dtype)], dtype
      )
      values = values[:, None]
      fn = lambda a, s, x: normal.Normal(a, s).log_prob(x)
      actual = (
          jax.jit(fn)(loc, scale, values)
          if compiled
          else fn(loc, scale, values)
      )
      expected = np.asarray(
          [[reference(a, scale, x)[0] for a in loc] for x in values[:, 0]]
      )
      self.assertEqual(actual.dtype, dtype)
      self.assertTrue(np.isfinite(np.asarray(actual)).all())
      np.testing.assert_allclose(
          np.asarray(actual, np.float64), expected, rtol=tolerance(dtype)
      )

  @parameterized.product(dtype=DTYPES, compiled=(False, True))
  def test_gradients(self, dtype, compiled):
    with x64_enabled(dtype == jnp.float64):
      args = [jnp.asarray(v, dtype) for v in (0.0, 2.0, 2 * residual(dtype))]
      fn = jax.grad(
          lambda a, s, x: normal.Normal(a, s).log_prob(x), argnums=(0, 1, 2)
      )
      actual = jax.jit(fn)(*args) if compiled else fn(*args)
      expected = reference(*args)[1]
      actual = np.asarray([float(g) for g in actual])
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(
          np.asarray(actual, np.float64), expected, rtol=tolerance(dtype)
      )

  @parameterized.product(dtype=DTYPES, compiled=(False, True))
  def test_unrepresentable_log_prob(self, dtype, compiled):
    with x64_enabled(dtype == jnp.float64):
      values = jnp.asarray([-4 * residual(dtype), 4 * residual(dtype)], dtype)
      fn = lambda x: normal.Normal(jnp.asarray(0.0, dtype), 1.0).log_prob(x)
      actual = jax.jit(fn)(values) if compiled else fn(values)
      self.assertTrue(np.isneginf(np.asarray(actual)).all())

  @parameterized.product(
      dtype=(jnp.float32, jnp.float64),
      compiled=(False, True),
      multivariate=(False, True),
  )
  def test_native_gaussian_callers(self, dtype, compiled, multivariate):
    with x64_enabled(dtype == jnp.float64):
      loc = jnp.zeros(2, dtype)
      scale = jnp.full(2, 2.0, dtype)
      values = jnp.asarray([2 * residual(dtype), 0.0], dtype)
      dist = (
          mvn_diag.MultivariateNormalDiag(loc, scale)
          if multivariate
          else (independent.Independent(normal.Normal(loc, scale), 1))
      )
      actual = (
          jax.jit(dist.log_prob)(values) if compiled else dist.log_prob(values)
      )
      expected = sum(
          reference(a, s, x)[0] for a, s, x in zip(loc, scale, values)
      )
      self.assertTrue(np.isfinite(float(actual)))
      np.testing.assert_allclose(actual, expected, rtol=tolerance(dtype))


if __name__ == '__main__':
  absltest.main()
