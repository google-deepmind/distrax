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
"""Tests for `bijector_from_tfp.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import bijector_from_tfp
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


class BijectorFromTFPTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    bjs = {}
    bjs['BatchedChain'] = tfb.Chain([
        tfb.Shift(jnp.zeros((4, 2, 3))),
        tfb.ScaleMatvecDiag([[1., 2., 3.], [4., 5., 6.]])
    ])
    bjs['Square'] = tfb.Square()
    bjs['ScaleScalar'] = tfb.Scale(2.)
    bjs['ScaleMatrix'] = tfb.Scale(2. * jnp.ones((3, 2)))
    bjs['Reshape'] = tfb.Reshape((2, 3), (6,))

    # To parallelize pytest runs.
    # See https://github.com/pytest-dev/pytest-xdist/issues/432.
    for name, bij in bjs.items():
      bij.__repr__ = lambda _, name_=name: name_

    self._test_bijectors = bjs

  @chex.all_variants
  @parameterized.parameters(
      ('Square', (), (), (), ()),
      ('Square', (2, 3), (), (2, 3), ()),
      ('ScaleScalar', (), (), (), ()),
      ('ScaleScalar', (2, 3), (), (2, 3), ()),
      ('ScaleMatrix', (), (), (3, 2), ()),
      ('ScaleMatrix', (2,), (), (3, 2), ()),
      ('ScaleMatrix', (1, 1), (), (3, 2), ()),
      ('ScaleMatrix', (4, 1, 1), (), (4, 3, 2), ()),
      ('ScaleMatrix', (4, 3, 2), (), (4, 3, 2), ()),
      ('Reshape', (), (6,), (), (2, 3)),
      ('Reshape', (10,), (6,), (10,), (2, 3)),
      ('BatchedChain', (), (3,), (4, 2), (3,)),
      ('BatchedChain', (2,), (3,), (4, 2), (3,)),
      ('BatchedChain', (4, 1), (3,), (4, 2), (3,)),
      ('BatchedChain', (5, 1, 2), (3,), (5, 4, 2), (3,)),
  )
  def test_forward_methods_are_correct(self, tfp_bij_name, batch_shape_in,
                                       event_shape_in, batch_shape_out,
                                       event_shape_out):
    tfp_bij = self._test_bijectors[tfp_bij_name]
    bij = bijector_from_tfp.BijectorFromTFP(tfp_bij)
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, batch_shape_in + event_shape_in)

    y = self.variant(bij.forward)(x)
    logdet = self.variant(bij.forward_log_det_jacobian)(x)
    y_tfp = tfp_bij.forward(x)
    logdet_tfp = tfp_bij.forward_log_det_jacobian(x, len(event_shape_in))
    logdet_tfp = jnp.broadcast_to(logdet_tfp, batch_shape_out)

    self.assertEqual(y.shape, batch_shape_out + event_shape_out)
    self.assertEqual(logdet.shape, batch_shape_out)
    np.testing.assert_allclose(y, y_tfp, atol=1e-8)
    np.testing.assert_allclose(logdet, logdet_tfp, atol=1e-4)

  @chex.all_variants
  @parameterized.parameters(
      ('Square', (), (), (), ()),
      ('Square', (2, 3), (), (2, 3), ()),
      ('ScaleScalar', (), (), (), ()),
      ('ScaleScalar', (2, 3), (), (2, 3), ()),
      ('ScaleMatrix', (3, 2), (), (), ()),
      ('ScaleMatrix', (3, 2), (), (2,), ()),
      ('ScaleMatrix', (3, 2), (), (1, 1), ()),
      ('ScaleMatrix', (4, 3, 2), (), (4, 1, 1), ()),
      ('ScaleMatrix', (4, 3, 2), (), (4, 3, 2), ()),
      ('Reshape', (), (6,), (), (2, 3)),
      ('Reshape', (10,), (6,), (10,), (2, 3)),
      ('BatchedChain', (4, 2), (3,), (), (3,)),
      ('BatchedChain', (4, 2), (3,), (2,), (3,)),
      ('BatchedChain', (4, 2), (3,), (4, 1), (3,)),
      ('BatchedChain', (5, 4, 2), (3,), (5, 1, 2), (3,)),
  )
  def test_inverse_methods_are_correct(self, tfp_bij_name, batch_shape_in,
                                       event_shape_in, batch_shape_out,
                                       event_shape_out):
    tfp_bij = self._test_bijectors[tfp_bij_name]
    bij = bijector_from_tfp.BijectorFromTFP(tfp_bij)
    key = jax.random.PRNGKey(42)
    y = jax.random.uniform(key, batch_shape_out + event_shape_out)

    x = self.variant(bij.inverse)(y)
    logdet = self.variant(bij.inverse_log_det_jacobian)(y)
    x_tfp = tfp_bij.inverse(y)
    logdet_tfp = tfp_bij.inverse_log_det_jacobian(y, len(event_shape_out))
    logdet_tfp = jnp.broadcast_to(logdet_tfp, batch_shape_in)

    self.assertEqual(x.shape, batch_shape_in + event_shape_in)
    self.assertEqual(logdet.shape, batch_shape_in)
    np.testing.assert_allclose(x, x_tfp, atol=1e-8)
    np.testing.assert_allclose(logdet, logdet_tfp, atol=1e-4)

  @chex.all_variants
  @parameterized.parameters(
      ('Square', (), (), (), ()),
      ('Square', (2, 3), (), (2, 3), ()),
      ('ScaleScalar', (), (), (), ()),
      ('ScaleScalar', (2, 3), (), (2, 3), ()),
      ('ScaleMatrix', (), (), (), ()),
      ('ScaleMatrix', (2,), (), (2,), ()),
      ('ScaleMatrix', (1, 1), (), (1, 1), ()),
      ('ScaleMatrix', (4, 1, 1), (), (4, 1, 1), ()),
      ('ScaleMatrix', (4, 3, 2), (), (4, 3, 2), ()),
      ('Reshape', (), (6,), (), (2, 3)),
      ('Reshape', (10,), (6,), (10,), (2, 3)),
      ('BatchedChain', (), (3,), (), (3,)),
      ('BatchedChain', (2,), (3,), (2,), (3,)),
      ('BatchedChain', (4, 1), (3,), (4, 1), (3,)),
      ('BatchedChain', (5, 1, 2), (3,), (5, 1, 2), (3,)),
  )
  def test_composite_methods_are_consistent(self, tfp_bij_name, batch_shape_in,
                                            event_shape_in, batch_shape_out,
                                            event_shape_out):
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    tfp_bij = self._test_bijectors[tfp_bij_name]
    bij = bijector_from_tfp.BijectorFromTFP(tfp_bij)

    # Forward methods.
    x = jax.random.uniform(key1, batch_shape_in + event_shape_in)
    y1 = self.variant(bij.forward)(x)
    logdet1 = self.variant(bij.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bij.forward_and_log_det)(x)
    self.assertEqual(y1.shape, y2.shape)
    self.assertEqual(logdet1.shape, logdet2.shape)
    np.testing.assert_allclose(y1, y2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-8)

    # Inverse methods.
    y = jax.random.uniform(key2, batch_shape_out + event_shape_out)
    x1 = self.variant(bij.inverse)(y)
    logdet1 = self.variant(bij.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bij.inverse_and_log_det)(y)
    self.assertEqual(x1.shape, x2.shape)
    self.assertEqual(logdet1.shape, logdet2.shape)
    np.testing.assert_allclose(x1, x2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-8)

  @chex.all_variants
  @parameterized.parameters(
      ('Square', (), (), (), ()),
      ('Square', (2, 3), (), (2, 3), ()),
      ('ScaleScalar', (), (), (), ()),
      ('ScaleScalar', (2, 3), (), (2, 3), ()),
      ('ScaleMatrix', (), (), (), ()),
      ('ScaleMatrix', (2,), (), (2,), ()),
      ('ScaleMatrix', (1, 1), (), (1, 1), ()),
      ('ScaleMatrix', (4, 1, 1), (), (4, 1, 1), ()),
      ('ScaleMatrix', (4, 3, 2), (), (4, 3, 2), ()),
      ('Reshape', (), (6,), (), (2, 3)),
      ('Reshape', (10,), (6,), (10,), (2, 3)),
      ('BatchedChain', (), (3,), (), (3,)),
      ('BatchedChain', (2,), (3,), (2,), (3,)),
      ('BatchedChain', (4, 1), (3,), (4, 1), (3,)),
      ('BatchedChain', (5, 1, 2), (3,), (5, 1, 2), (3,)),
  )
  def test_works_with_tfp_caching(self, tfp_bij_name, batch_shape_in,
                                  event_shape_in, batch_shape_out,
                                  event_shape_out):
    tfp_bij = self._test_bijectors[tfp_bij_name]
    bij = bijector_from_tfp.BijectorFromTFP(tfp_bij)
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))

    # Forward caching.
    x = jax.random.uniform(key1, batch_shape_in + event_shape_in)
    y = self.variant(bij.forward)(x)
    x1 = self.variant(bij.inverse)(y)
    logdet1 = self.variant(bij.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bij.inverse_and_log_det)(y)
    self.assertEqual(x1.shape, x2.shape)
    self.assertEqual(logdet1.shape, logdet2.shape)
    np.testing.assert_allclose(x1, x2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-8)

    # Inverse caching.
    y = jax.random.uniform(key2, batch_shape_out + event_shape_out)
    x = self.variant(bij.inverse)(y)
    y1 = self.variant(bij.forward)(x)
    logdet1 = self.variant(bij.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bij.forward_and_log_det)(x)
    self.assertEqual(y1.shape, y2.shape)
    self.assertEqual(logdet1.shape, logdet2.shape)
    np.testing.assert_allclose(y1, y2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-8)

  def test_access_properties_tfp_bijector(self):
    tfp_bij = self._test_bijectors['BatchedChain']
    bij = bijector_from_tfp.BijectorFromTFP(tfp_bij)
    # Access the attribute `bijectors`
    np.testing.assert_allclose(
        bij.bijectors[0].shift, tfp_bij.bijectors[0].shift, atol=1e-8)
    np.testing.assert_allclose(
        bij.bijectors[1].scale.diag, tfp_bij.bijectors[1].scale.diag, atol=1e-8)

  def test_jittable(self):

    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = bijector_from_tfp.BijectorFromTFP(tfb.Tanh())
    x = np.zeros(())
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
