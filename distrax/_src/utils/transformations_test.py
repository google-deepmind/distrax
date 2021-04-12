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
"""Tests for `transformations.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.utils import transformations
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


RTOL = 1e-2


class TransformationsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  @chex.all_variants
  @parameterized.named_parameters(
      ('tanh', jnp.tanh, jnp.arctanh, 0.5),
      ('arctanh', jnp.arctanh, jnp.tanh, 0.5),
      ('sinh', jnp.sinh, jnp.arcsinh, 0.5),
      ('arcsinh', jnp.arcsinh, jnp.sinh, 0.5),
      ('cosh', jnp.cosh, jnp.arccosh, 0.5),
      ('arccosh', jnp.arccosh, jnp.cosh, 2.0),
      ('exp', jnp.exp, jnp.log, 0.5),
      ('log', jnp.log, jnp.exp, 0.5),
      ('pow', lambda x: jnp.power(x, 3.5), lambda y: jnp.power(y, 1/3.5), 0.5),
      ('add', lambda x: x + 3, lambda y: y - 3, 0.5),
      ('sqrt', jnp.sqrt, jnp.square, 0.5),
      ('square', jnp.square, jnp.sqrt, 0.5),
      ('reciprocal', jnp.reciprocal, jnp.reciprocal, 0.5),
      ('negate', lambda x: -x, lambda y: -y, 0.5),
      ('log1p', jnp.log1p, jnp.expm1, 0.5),
      ('expm1', jnp.expm1, jnp.log1p, 0.5),
      ('erf', jax.lax.erf, jax.lax.erf_inv, 0.5),
      ('erf_inv', jax.lax.erf_inv, jax.lax.erf, 0.5),
      ('2_mul_x', lambda x: 2 * x, lambda y: y * 0.5, 0.5),
      ('x_mul_2', lambda x: x * 2, lambda y: y * 0.5, 0.5),
      ('2_div_x', lambda x: 2 / x, lambda y: 2 / y, 0.5),
      ('x_div_2', lambda x: x / 2, lambda y: y / 0.5, 0.5),
      ('x_sub_3', lambda x: x - 3, lambda y: y + 3, 0.5),
      ('3_sub_x', lambda x: 3 - x, lambda y: 3 - y, 0.5),
      ('x**3.5', lambda x: x**3.5, lambda y: y**(1/3.5), 0.5),
      ('x**(1/3.5)', lambda x: x**(1/3.5), lambda y: y**3.5, 0.5),
  )
  def test_inversion(self, forward, inverse, x):
    x = jnp.array([x], dtype=jnp.float32)
    y = forward(x)

    inverse_ = self.variant(transformations.inv(forward))
    x_ = inverse_(y)

    np.testing.assert_allclose(x_, x, rtol=RTOL)
    np.testing.assert_allclose(x_, inverse(y), rtol=RTOL)

  @chex.all_variants
  @parameterized.named_parameters(
      ('tanh', jnp.tanh, jnp.arctanh, 0.5),
      ('arctanh', jnp.arctanh, jnp.tanh, 0.5),
      ('sinh', jnp.sinh, jnp.arcsinh, 0.5),
      ('arcsinh', jnp.arcsinh, jnp.sinh, 0.5),
      ('cosh', jnp.cosh, jnp.arccosh, 0.5),
      ('arccosh', jnp.arccosh, jnp.cosh, 2.0),
      ('exp', jnp.exp, jnp.log, 0.5),
      ('log', jnp.log, jnp.exp, 0.5),
      ('pow', lambda x: jnp.power(x, 3.5), lambda y: jnp.power(y, 1/3.5), 0.5),
      ('add', lambda x: x + 3, lambda y: y - 3, 0.5),
      ('sqrt', jnp.sqrt, jnp.square, 0.5),
      ('square', jnp.square, jnp.sqrt, 0.5),
      ('reciprocal', jnp.reciprocal, jnp.reciprocal, 0.5),
      ('negate', lambda x: -x, lambda y: -y, 0.5),
      ('log1p', jnp.log1p, jnp.expm1, 0.5),
      ('expm1', jnp.expm1, jnp.log1p, 0.5),
      ('erf', jax.lax.erf, jax.lax.erf_inv, 0.5),
      ('erf_inv', jax.lax.erf_inv, jax.lax.erf, 0.5),
      ('2_mul_x', lambda x: 2 * x, lambda y: y * 0.5, 0.5),
      ('x_mul_2', lambda x: x * 2, lambda y: y * 0.5, 0.5),
      ('2_div_x', lambda x: 2 / x, lambda y: 2 / y, 0.5),
      ('x_div_2', lambda x: x / 2, lambda y: y / 0.5, 0.5),
      ('x_sub_3', lambda x: x - 3, lambda y: y + 3, 0.5),
      ('3_sub_x', lambda x: 3 - x, lambda y: 3 - y, 0.5),
      ('x**3.5', lambda x: x**3.5, lambda y: y**(1/3.5), 0.5),
      ('x**(1/3.5)', lambda x: x**(1/3.5), lambda y: y**3.5, 0.5),
  )
  def test_inverting_jitted_function(self, forward, inverse, x):
    x = jnp.array([x], dtype=jnp.float32)
    y = forward(x)

    jitted_forward = jax.jit(forward)
    inverse_ = self.variant(transformations.inv(jitted_forward))
    x_ = inverse_(y)

    np.testing.assert_allclose(x_, x, rtol=RTOL)
    np.testing.assert_allclose(x_, inverse(y), rtol=RTOL)

  @chex.all_variants
  @parameterized.named_parameters(
      ('identity, 0d', lambda x: x, tfb.Identity, 0.5),
      ('identity, 1d', lambda x: x, tfb.Identity, [0.9]),
      ('identity, 2d', lambda x: x, tfb.Identity, [0.25, 0.75]),
      ('identity, 2x2d', lambda x: x, tfb.Identity, [[0.25, 0.75],
                                                     [0.1, 0.9]]),
      ('scale, 0d', lambda x: 3.0 * x, lambda: tfb.Scale(3.0), 0.5),
      ('scale, 1d', lambda x: 3.0 * x, lambda: tfb.Scale(3.0), [0.9]),
      ('scale, 2d', lambda x: 3.0 * x, lambda: tfb.Scale(3.0), [0.25, 0.75]),
      ('scale, 2x2d', lambda x: 3.0 * x, lambda: tfb.Scale(3.0), [[0.25, 0.75],
                                                                  [0.1, 0.9]]),
      ('tanh, 0d', jnp.tanh, tfb.Tanh, 0.5),
      ('tanh, 1d', jnp.tanh, tfb.Tanh, [0.9]),
      ('tanh, 2d', jnp.tanh, tfb.Tanh, [0.25, 0.75]),
      ('tanh, 2x2d', jnp.tanh, tfb.Tanh, [[0.25, 0.75],
                                          [0.1, 0.9]]),
      ('softplus, 0d', jax.nn.softplus, tfb.Softplus, 0.5),
      ('softplus, 1d', jax.nn.softplus, tfb.Softplus, [0.9]),
      ('softplus, 2d', jax.nn.softplus, tfb.Softplus, [0.25, 0.75]),
      ('softplus, 2x2d', jax.nn.softplus, tfb.Softplus, [[0.25, 0.75],
                                                         [0.1, 0.9]]),
      ('sigmoid, 0d', jax.nn.sigmoid, tfb.Sigmoid, 0.5),
      ('sigmoid, 1d', jax.nn.sigmoid, tfb.Sigmoid, [0.9]),
      ('sigmoid, 2d', jax.nn.sigmoid, tfb.Sigmoid, [0.25, 0.75]),
      ('sigmoid, 2x2d', jax.nn.sigmoid, tfb.Sigmoid, [[0.25, 0.75],
                                                      [0.1, 0.9]]),
  )
  def test_log_det_scalar(self, forward, tfb_bijector, x):
    x = np.array(x, dtype=np.float32)

    log_det_fn = self.variant(transformations.log_det_scalar(forward))

    actual = log_det_fn(x)
    expected = tfb_bijector().forward_log_det_jacobian(x, event_ndims=0)

    np.testing.assert_allclose(actual, expected, rtol=RTOL)

  @parameterized.named_parameters(
      ('tanh', jnp.tanh, False),
      ('sigmoid', jax.nn.sigmoid, False),
      ('identity', lambda x: x, True),
      ('square', lambda x: x**2, False),
      ('softplus', jax.nn.softplus, False),
      ('exp', jnp.exp, False),
      ('log', jnp.log, False, 1.0),
      ('shift', lambda x: x + 3.0, True),
      ('scale', lambda x: 2.0 * x, True),
      ('shift and scale', lambda x: 2.0 * x + 3.0, True),
  )
  def test_is_constant_jacobian(self, fn, is_constant, x=0.0):
    is_constant_ = transformations.is_constant_jacobian(fn, x)
    np.testing.assert_array_equal(is_constant, is_constant_)


if __name__ == '__main__':
  absltest.main()
