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
"""Tests for `rational_quadratic_spline.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import rational_quadratic_spline
import jax
import jax.numpy as jnp
import numpy as np


def _make_bijector(params_shape,
                   zero_params=False,
                   num_bins=8,
                   range_min=0.,
                   range_max=1.,
                   boundary_slopes='unconstrained'):
  """Returns RationalQuadraticSpline bijector."""
  params_shape += (3 * num_bins + 1,)
  if zero_params:
    params = jnp.zeros(params_shape)
  else:
    key = jax.random.PRNGKey(101)
    params = jax.random.normal(key, params_shape)
  return rational_quadratic_spline.RationalQuadraticSpline(
      params,
      range_min=range_min,
      range_max=range_max,
      boundary_slopes=boundary_slopes)


class RationalQuadraticSplineTest(parameterized.TestCase):
  """Tests for rational quadratic spline."""

  def test_properties(self):
    bijector = _make_bijector(params_shape=(4, 5), num_bins=8)
    self.assertEqual(bijector.is_constant_jacobian, False)
    self.assertEqual(bijector.is_constant_log_det, False)
    assert bijector.num_bins == 8
    self.assertEqual(bijector.knot_slopes.shape, (4, 5, 9))
    self.assertEqual(bijector.x_pos.shape, (4, 5, 9))
    self.assertEqual(bijector.y_pos.shape, (4, 5, 9))

  @parameterized.named_parameters(
      ('params.shape[-1] < 4',
       {'params': np.zeros((3,)), 'range_min': 0., 'range_max': 1.}),
      ('params.shape[-1] % 3 is not 1',
       {'params': np.zeros((8,)), 'range_min': 0., 'range_max': 1.}),
      ('inconsistent range min and max',
       {'params': np.zeros((10,)), 'range_min': 1., 'range_max': 0.9}),
      ('negative min_bin_size',
       {'params': np.zeros((10,)), 'range_min': 0., 'range_max': 1.,
        'min_bin_size': -0.1}),
      ('negative min_knot_slope',
       {'params': np.zeros((10,)), 'range_min': 0., 'range_max': 1.,
        'min_knot_slope': -0.1}),
      ('min_knot_slope above 1',
       {'params': np.zeros((10,)), 'range_min': 0., 'range_max': 1.,
        'min_knot_slope': 1.3}),
      ('invalid boundary_slopes',
       {'params': np.zeros((10,)), 'range_min': 0., 'range_max': 1.,
        'boundary_slopes': 'invalid_value'}),
      ('num_bins * min_bin_size greater than total_size',
       {'params': np.zeros((10,)), 'range_min': 0., 'range_max': 1.,
        'min_bin_size': 0.9}),
  )
  def test_invalid_properties(self, bij_params):
    with self.assertRaises(ValueError):
      rational_quadratic_spline.RationalQuadraticSpline(**bij_params)

  @chex.all_variants
  def test_shapes_are_correct(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _make_bijector(params_shape=(4, 5))
    # Forward methods.
    y, logdet = self.variant(bijector.forward_and_log_det)(x)
    self.assertEqual(y.shape, (2, 3, 4, 5))
    self.assertEqual(logdet.shape, (2, 3, 4, 5))
    # Inverse methods.
    x, logdet = self.variant(bijector.inverse_and_log_det)(y)
    self.assertEqual(x.shape, (2, 3, 4, 5))
    self.assertEqual(logdet.shape, (2, 3, 4, 5))

  @chex.all_variants
  def test_broadcasting_is_correct(self):
    z = 0.5 * jnp.ones((2, 2))
    # Broadcast along first axis.
    bijector = _make_bijector(params_shape=(2,))
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(z)
    x, logdet_inv = self.variant(bijector.inverse_and_log_det)(z)
    np.testing.assert_array_equal(y[0], y[1])
    np.testing.assert_array_equal(x[0], x[1])
    np.testing.assert_array_equal(logdet_fwd[0], logdet_fwd[1])
    np.testing.assert_array_equal(logdet_inv[0], logdet_inv[1])
    self.assertFalse(jnp.allclose(y[:, 0], y[:, 1]))
    self.assertFalse(jnp.allclose(x[:, 0], x[:, 1]))
    self.assertFalse(jnp.allclose(logdet_fwd[:, 0], logdet_fwd[:, 1]))
    self.assertFalse(jnp.allclose(logdet_inv[:, 0], logdet_inv[:, 1]))
    # Broadcast along second axis.
    bijector = _make_bijector(params_shape=(2, 1))
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(z)
    x, logdet_inv = self.variant(bijector.inverse_and_log_det)(z)
    np.testing.assert_array_equal(y[:, 0], y[:, 1])
    np.testing.assert_array_equal(x[:, 0], x[:, 1])
    np.testing.assert_array_equal(logdet_fwd[:, 0], logdet_fwd[:, 1])
    np.testing.assert_array_equal(logdet_inv[:, 0], logdet_inv[:, 1])
    self.assertFalse(jnp.allclose(y[0], y[1]))
    self.assertFalse(jnp.allclose(x[0], x[1]))
    self.assertFalse(jnp.allclose(logdet_fwd[0], logdet_fwd[1]))
    self.assertFalse(jnp.allclose(logdet_inv[0], logdet_inv[1]))

  @chex.all_variants
  def test_is_identity_for_zero_params(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _make_bijector(params_shape=(4, 5), zero_params=True)
    # Forward methods.
    y, logdet = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y, x, atol=5e-5)
    np.testing.assert_allclose(logdet, jnp.zeros((2, 3, 4, 5)), atol=5e-5)
    # Inverse methods.
    x, logdet = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(y, x, atol=5e-5)
    np.testing.assert_allclose(logdet, jnp.zeros((2, 3, 4, 5)), atol=5e-5)

  @chex.all_variants
  def test_inverse_methods_are_correct(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _make_bijector(params_shape=(4, 5))
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    x_rec, logdet_inv = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_rec, x, atol=7e-4)
    np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=7e-4)

  @chex.all_variants
  def test_is_monotonically_increasing(self):
    z = jnp.linspace(start=-2, stop=2, num=100)
    bijector = _make_bijector(params_shape=())
    y = self.variant(bijector.forward)(z)
    x = self.variant(bijector.inverse)(z)
    np.testing.assert_array_less(y[:-1], y[1:])
    np.testing.assert_array_less(x[:-1], x[1:])

  @chex.all_variants
  def test_composite_methods_are_consistent(self):
    key = jax.random.PRNGKey(42)
    bijector = _make_bijector(params_shape=(4, 5))
    # Forward methods.
    x = jax.random.normal(key, (2, 3, 4, 5))
    y1 = self.variant(bijector.forward)(x)
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y1, y2, atol=7e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=7e-8)
    # Inverse methods.
    y = jax.random.normal(key, (2, 3, 4, 5))
    x1 = self.variant(bijector.inverse)(y)
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x1, x2, atol=7e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=7e-8)

  @chex.all_variants
  def test_boundary_conditions(self):
    a = jnp.array(0.)
    b = jnp.array(1.)
    # Unconstrained boundary slopes.
    bijector = _make_bijector(
        params_shape=(),
        range_min=float(a),
        range_max=float(b),
        boundary_slopes='unconstrained')
    log_slope_a = self.variant(bijector.forward_log_det_jacobian)(a)
    log_slope_b = self.variant(bijector.forward_log_det_jacobian)(b)
    self.assertEqual(self.variant(bijector.forward)(a), a)
    self.assertEqual(self.variant(bijector.forward)(b), b)
    self.assertFalse(jnp.allclose(log_slope_a, 0.))
    self.assertFalse(jnp.allclose(log_slope_b, 0.))
    # Lower boundary slope equal to 1.
    bijector = _make_bijector(
        params_shape=(),
        range_min=float(a),
        range_max=float(b),
        boundary_slopes='lower_identity')
    log_slope_a = self.variant(bijector.forward_log_det_jacobian)(a)
    log_slope_b = self.variant(bijector.forward_log_det_jacobian)(b)
    self.assertEqual(self.variant(bijector.forward)(a), a)
    self.assertEqual(self.variant(bijector.forward)(b), b)
    self.assertEqual(log_slope_a, 0.)
    self.assertFalse(jnp.allclose(log_slope_b, 0.))
    # Upper boundary slope equal to 1.
    bijector = _make_bijector(
        params_shape=(),
        range_min=float(a),
        range_max=float(b),
        boundary_slopes='upper_identity')
    log_slope_a = self.variant(bijector.forward_log_det_jacobian)(a)
    log_slope_b = self.variant(bijector.forward_log_det_jacobian)(b)
    self.assertEqual(self.variant(bijector.forward)(a), a)
    self.assertEqual(self.variant(bijector.forward)(b), b)
    self.assertFalse(jnp.allclose(log_slope_a, 0.))
    self.assertEqual(log_slope_b, 0.)
    # Both boundary slopes equal to 1.
    bijector = _make_bijector(
        params_shape=(),
        range_min=float(a),
        range_max=float(b),
        boundary_slopes='identity')
    log_slope_a = self.variant(bijector.forward_log_det_jacobian)(a)
    log_slope_b = self.variant(bijector.forward_log_det_jacobian)(b)
    self.assertEqual(self.variant(bijector.forward)(a), a)
    self.assertEqual(self.variant(bijector.forward)(b), b)
    self.assertEqual(log_slope_a, 0.)
    self.assertEqual(log_slope_b, 0.)
    # Circular spline (periodic slope).
    bijector = _make_bijector(
        params_shape=(),
        range_min=float(a),
        range_max=float(b),
        boundary_slopes='circular')
    log_slope_a = self.variant(bijector.forward_log_det_jacobian)(a)
    log_slope_b = self.variant(bijector.forward_log_det_jacobian)(b)
    self.assertEqual(self.variant(bijector.forward)(a), a)
    self.assertEqual(self.variant(bijector.forward)(b), b)
    self.assertEqual(log_slope_a, log_slope_b)
    self.assertFalse(jnp.allclose(log_slope_b, 0.))

  @chex.all_variants
  @parameterized.parameters(
      ((3, 4), (3, 4)),
      ((3, 4), (3, 1)),
      ((3, 4), (4,)),
      ((3, 4), ()),
      ((3, 1), (3, 4)),
      ((4,), (3, 4)),
      ((), (3, 4)),
  )
  def test_batched_parameters(self, params_batch_shape, input_batch_shape):
    k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
    num_bins = 4
    param_dim = 3 * num_bins + 1
    params = jax.random.normal(k1, params_batch_shape + (param_dim,))
    bijector = rational_quadratic_spline.RationalQuadraticSpline(
        params, range_min=0., range_max=1.)

    x = jax.random.uniform(k2, input_batch_shape)
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    z, logdet_inv = self.variant(bijector.inverse_and_log_det)(x)

    output_batch_shape = jnp.broadcast_arrays(params[..., 0], x)[0].shape

    self.assertEqual(y.shape, output_batch_shape)
    self.assertEqual(z.shape, output_batch_shape)
    self.assertEqual(logdet_fwd.shape, output_batch_shape)
    self.assertEqual(logdet_inv.shape, output_batch_shape)

    params = jnp.broadcast_to(
        params, output_batch_shape + (param_dim,)).reshape((-1, param_dim))
    x = jnp.broadcast_to(x, output_batch_shape).flatten()
    y = y.flatten()
    z = z.flatten()
    logdet_fwd = logdet_fwd.flatten()
    logdet_inv = logdet_inv.flatten()

    for i in range(np.prod(output_batch_shape)):
      bijector = rational_quadratic_spline.RationalQuadraticSpline(
          params[i], range_min=0., range_max=1.)
      this_y, this_logdet_fwd = self.variant(bijector.forward_and_log_det)(x[i])
      this_z, this_logdet_inv = self.variant(bijector.inverse_and_log_det)(x[i])
      np.testing.assert_allclose(this_y, y[i], atol=1e-7)
      np.testing.assert_allclose(this_z, z[i], atol=1e-6)
      np.testing.assert_allclose(this_logdet_fwd, logdet_fwd[i], atol=1e-5)
      np.testing.assert_allclose(this_logdet_inv, logdet_inv[i], atol=1e-5)

  @chex.all_variants
  @parameterized.parameters(
      (-1., 4., -3., 1.,),  # when b >= 0
      (1., -4., 3., 3.),  # when b < 0
      (-1., 2., -1., 1.),  # when b**2 - 4*a*c = 0, and b >= 0
      (1., -2., 1., 1.),  # when b**2 - 4*a*c = 0, and b < 0
  )
  def test_safe_quadratic_root(self, a, b, c, x):
    a = jnp.array(a)
    b = jnp.array(b)
    c = jnp.array(c)
    x = jnp.array(x)
    sol_x, grad = self.variant(jax.value_and_grad(
        rational_quadratic_spline._safe_quadratic_root))(a, b, c)
    np.testing.assert_allclose(sol_x, x, atol=1e-5)
    self.assertFalse(np.any(np.isnan(grad)))

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = _make_bijector(params_shape=(4, 5))
    x = np.zeros((2, 3, 4, 5))
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
