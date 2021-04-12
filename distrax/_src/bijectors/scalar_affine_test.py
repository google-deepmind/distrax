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
"""Tests for `scalar_affine.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import scalar_affine
import jax
import jax.numpy as jnp
import numpy as np


class ScalarAffineTest(parameterized.TestCase):

  def test_properties(self):
    bij = scalar_affine.ScalarAffine(shift=0., scale=1.)
    self.assertTrue(bij.is_constant_jacobian)
    self.assertTrue(bij.is_constant_log_det)
    np.testing.assert_allclose(bij.shift, 0.)
    np.testing.assert_allclose(bij.scale, 1.)
    np.testing.assert_allclose(bij.log_scale, 0.)

  def test_raises_if_both_scale_and_log_scale_are_specified(self):
    with self.assertRaises(ValueError):
      scalar_affine.ScalarAffine(shift=0., scale=1., log_scale=0.)

  @chex.all_variants
  def test_shapes_are_correct(self):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    x = jax.random.normal(k1, (2, 3, 4, 5))
    shift = jax.random.normal(k2, (4, 5))
    scale = jax.random.uniform(k3, (3, 4, 5)) + 0.1
    log_scale = jax.random.normal(k4, (3, 4, 5))
    bij_no_scale = scalar_affine.ScalarAffine(shift)
    bij_with_scale = scalar_affine.ScalarAffine(shift, scale=scale)
    bij_with_log_scale = scalar_affine.ScalarAffine(shift, log_scale=log_scale)
    for bij in [bij_no_scale, bij_with_scale, bij_with_log_scale]:
      # Forward methods.
      y, logdet = self.variant(bij.forward_and_log_det)(x)
      self.assertEqual(y.shape, (2, 3, 4, 5))
      self.assertEqual(logdet.shape, (2, 3, 4, 5))
      # Inverse methods.
      x, logdet = self.variant(bij.inverse_and_log_det)(y)
      self.assertEqual(x.shape, (2, 3, 4, 5))
      self.assertEqual(logdet.shape, (2, 3, 4, 5))

  @chex.all_variants
  def test_forward_methods_are_correct(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bij_no_scale = scalar_affine.ScalarAffine(shift=3.)
    bij_with_scale = scalar_affine.ScalarAffine(shift=3., scale=1.)
    bij_with_log_scale = scalar_affine.ScalarAffine(shift=3., log_scale=0.)
    for bij in [bij_no_scale, bij_with_scale, bij_with_log_scale]:
      y, logdet = self.variant(bij.forward_and_log_det)(x)
      np.testing.assert_allclose(y, x + 3., atol=1e-8)
      np.testing.assert_allclose(logdet, 0., atol=1e-8)

  @chex.all_variants
  def test_inverse_methods_are_correct(self):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    x = jax.random.normal(k1, (2, 3, 4, 5))
    shift = jax.random.normal(k2, (4, 5))
    scale = jax.random.uniform(k3, (3, 4, 5)) + 0.1
    log_scale = jax.random.normal(k4, (3, 4, 5))
    bij_no_scale = scalar_affine.ScalarAffine(shift)
    bij_with_scale = scalar_affine.ScalarAffine(shift, scale=scale)
    bij_with_log_scale = scalar_affine.ScalarAffine(shift, log_scale=log_scale)
    for bij in [bij_no_scale, bij_with_scale, bij_with_log_scale]:
      y, logdet_fwd = self.variant(bij.forward_and_log_det)(x)
      x_rec, logdet_inv = self.variant(bij.inverse_and_log_det)(y)
      np.testing.assert_allclose(x_rec, x, atol=1e-5)
      np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=3e-6)

  @chex.all_variants
  def test_composite_methods_are_consistent(self):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    bij = scalar_affine.ScalarAffine(
        shift=jax.random.normal(k1, (4, 5)),
        log_scale=jax.random.normal(k2, (4, 5)))
    # Forward methods.
    x = jax.random.normal(k3, (2, 3, 4, 5))
    y1 = self.variant(bij.forward)(x)
    logdet1 = self.variant(bij.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bij.forward_and_log_det)(x)
    np.testing.assert_allclose(y1, y2, atol=1e-12)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-12)
    # Inverse methods.
    y = jax.random.normal(k4, (2, 3, 4, 5))
    x1 = self.variant(bij.inverse)(y)
    logdet1 = self.variant(bij.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bij.inverse_and_log_det)(y)
    np.testing.assert_allclose(x1, x2, atol=1e-12)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-12)

  @chex.all_variants
  @parameterized.parameters(
      ((5,), (5,), (5,)),
      ((5,), (5,), ()),
      ((5,), (), (5,)),
      ((), (5,), (5,)),
      ((), (), (5,)),
      ((), (5,), ()),
      ((5,), (), ()),
  )
  def test_batched_parameters(self, scale_batch_shape, shift_batch_shape,
                              input_batch_shape):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    log_scale = jax.random.normal(k1, scale_batch_shape)
    shift = jax.random.normal(k2, shift_batch_shape)
    bijector = scalar_affine.ScalarAffine(shift, log_scale=log_scale)

    x = jax.random.normal(k3, input_batch_shape)
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    z, logdet_inv = self.variant(bijector.inverse_and_log_det)(x)

    output_batch_shape = jnp.broadcast_arrays(log_scale, shift, x)[0].shape

    self.assertEqual(y.shape, output_batch_shape)
    self.assertEqual(z.shape, output_batch_shape)
    self.assertEqual(logdet_fwd.shape, output_batch_shape)
    self.assertEqual(logdet_inv.shape, output_batch_shape)

    log_scale = jnp.broadcast_to(log_scale, output_batch_shape).flatten()
    shift = jnp.broadcast_to(shift, output_batch_shape).flatten()
    x = jnp.broadcast_to(x, output_batch_shape).flatten()
    y = y.flatten()
    z = z.flatten()
    logdet_fwd = logdet_fwd.flatten()
    logdet_inv = logdet_inv.flatten()

    for i in range(np.prod(output_batch_shape)):
      bijector = scalar_affine.ScalarAffine(shift[i], jnp.exp(log_scale[i]))
      this_y, this_logdet_fwd = self.variant(bijector.forward_and_log_det)(x[i])
      this_z, this_logdet_inv = self.variant(bijector.inverse_and_log_det)(x[i])
      np.testing.assert_allclose(this_y, y[i], atol=1e-7)
      np.testing.assert_allclose(this_z, z[i], atol=1e-5)
      np.testing.assert_allclose(this_logdet_fwd, logdet_fwd[i], atol=1e-4)
      np.testing.assert_allclose(this_logdet_inv, logdet_inv[i], atol=1e-4)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = scalar_affine.ScalarAffine(0, 1)
    x = np.zeros(())
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
