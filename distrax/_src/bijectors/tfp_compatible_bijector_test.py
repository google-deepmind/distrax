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
"""Tests for `tfp_compatible_bijector.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors.block import Block
from distrax._src.bijectors.chain import Chain
from distrax._src.bijectors.lambda_bijector import Lambda
from distrax._src.bijectors.scalar_affine import ScalarAffine
from distrax._src.bijectors.tanh import Tanh
from distrax._src.bijectors.tfp_compatible_bijector import tfp_compatible_bijector
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

RTOL = 3e-3


class TFPCompatibleBijectorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('lambda identity, 0d sample', lambda: Block(lambda x: x, 1),
       tfb.Identity, ()),
      ('lambda identity, 1d sample', lambda: Block(lambda x: x, 1),
       tfb.Identity, (5,)),
      ('lambda identity, 2d sample', lambda: Block(lambda x: x, 1),
       tfb.Identity, (7, 5)),
      ('ScalarAffine, 0d sample',
       lambda: Block(ScalarAffine(3, 2), 1),
       lambda: tfb.Chain([tfb.Shift(3), tfb.Scale(2)]), ()),
      ('ScalarAffine, 1d sample',
       lambda: Block(ScalarAffine(3, 2), 1),
       lambda: tfb.Chain([tfb.Shift(3), tfb.Scale(2)]), (5,)),
      ('ScalarAffine, 2d sample',
       lambda: Block(ScalarAffine(3, 2), 1),
       lambda: tfb.Chain([tfb.Shift(3), tfb.Scale(2)]), (7, 5)),
      ('Tanh, 0d sample', lambda: Block(Tanh(), 1), tfb.Tanh, ()),
      ('Tanh, 1d sample', lambda: Block(Tanh(), 1), tfb.Tanh, (5,)),
      ('Tanh, 2d sample',
       lambda: Block(Tanh(), 1), tfb.Tanh, (7, 5)),
      ('Chain(Tanh, ScalarAffine), 0d sample',
       lambda: Block(Chain([Tanh(), ScalarAffine(0.2, 1.2)]), 1),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]), ()),
      ('Chain(Tanh, ScalarAffine), 1d sample',
       lambda: Block(Chain([Tanh(), ScalarAffine(0.2, 1.2)]), 1),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]), (5,)),
      ('Chain(Tanh, ScalarAffine), 2d sample',
       lambda: Block(Chain([Tanh(), ScalarAffine(0.2, 1.2)]), 1),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]), (3, 5)),
  )
  def test_transformed_distribution(
      self, dx_bijector_fn, tfp_bijector_fn, sample_shape):
    base_dist = tfd.MultivariateNormalDiag(np.zeros((3, 2)), np.ones((3, 2)))
    dx_bijector = dx_bijector_fn()
    wrapped_bijector = tfp_compatible_bijector(dx_bijector)
    tfp_bijector = tfp_bijector_fn()
    dist_with_wrapped = tfd.TransformedDistribution(base_dist, wrapped_bijector)
    dist_tfp_only = tfd.TransformedDistribution(base_dist, tfp_bijector)

    with self.subTest('sample'):
      dist_with_wrapped.sample(
          seed=jax.random.PRNGKey(0), sample_shape=sample_shape)

    with self.subTest('log_prob'):
      y = dist_tfp_only.sample(
          seed=jax.random.PRNGKey(0), sample_shape=sample_shape)
      log_prob_wrapped = dist_with_wrapped.log_prob(y)
      log_prob_tfp_only = dist_tfp_only.log_prob(y)
      np.testing.assert_allclose(log_prob_wrapped, log_prob_tfp_only, rtol=RTOL)

  @parameterized.named_parameters(
      ('identity', lambda: Lambda(lambda x: x, is_constant_jacobian=True),
       tfb.Identity, np.array([1, 1.5, 2], dtype=np.float32)),
      ('affine', lambda: ScalarAffine(np.ones(3, dtype=np.float32),  # pylint: disable=g-long-lambda
                                      np.full(3, 5.5, dtype=np.float32)),
       lambda: tfb.Chain([tfb.Shift(np.ones(3, dtype=np.float32)),  # pylint: disable=g-long-lambda
                          tfb.Scale(np.full(3, 5.5, dtype=np.float32))]),
       np.array([1, 1.5, 2], dtype=np.float32)),
      ('tanh', Tanh, tfb.Tanh, np.array([-0.1, 0.01, 0.1], dtype=np.float32)),
      ('chain(tanh, affine)', lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]),
       np.array([-0.1, 0.01, 0.1], dtype=np.float32))
  )
  def test_chain(self, dx_bijector_fn, tfb_bijector_fn, event):
    dx_bij = tfp_compatible_bijector(dx_bijector_fn())
    tfp_bij = tfb_bijector_fn()

    chain_with_dx = tfb.Chain([tfb.Shift(1.0), tfb.Scale(3.0), dx_bij])
    chain_with_tfp = tfb.Chain([tfb.Shift(1.0), tfb.Scale(3.0), tfp_bij])

    with self.subTest('forward'):
      y_dx = chain_with_dx.forward(event)
      y_tfp = chain_with_tfp.forward(event)
      np.testing.assert_allclose(y_dx, y_tfp, rtol=RTOL)

    with self.subTest('inverse'):
      y = chain_with_tfp.forward(event)
      x_dx = chain_with_dx.inverse(y)
      np.testing.assert_allclose(x_dx, event, rtol=RTOL)

  @parameterized.named_parameters(
      ('identity', lambda: Lambda(lambda x: x, is_constant_jacobian=True),
       tfb.Identity, np.array([1, 1.5, 2], dtype=np.float32)),
      ('affine', lambda: ScalarAffine(np.ones(3, dtype=np.float32),  # pylint: disable=g-long-lambda
                                      np.full(3, 5.5, dtype=np.float32)),
       lambda: tfb.Chain([tfb.Shift(np.ones(3, dtype=np.float32)),  # pylint: disable=g-long-lambda
                          tfb.Scale(np.full(3, 5.5, dtype=np.float32))]),
       np.array([1, 1.5, 2], dtype=np.float32)),
      ('tanh', Tanh, tfb.Tanh, np.array([-0.1, 0.01, 0.1], dtype=np.float32)),
      ('chain(tanh, affine)', lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]),
       np.array([-0.1, 0.01, 0.1], dtype=np.float32)),
  )
  def test_invert(self, dx_bijector_fn, tfb_bijector_fn, event):
    dx_bij = tfp_compatible_bijector(dx_bijector_fn())
    tfp_bij = tfb_bijector_fn()

    invert_with_dx = tfb.Invert(dx_bij)
    invert_with_tfp = tfb.Invert(tfp_bij)

    with self.subTest('forward'):
      y_dx = invert_with_dx.forward(event)
      y_tfp = invert_with_tfp.forward(event)
      np.testing.assert_allclose(y_dx, y_tfp, rtol=RTOL)

    with self.subTest('inverse'):
      y = invert_with_tfp.forward(event)
      x_dx = invert_with_dx.inverse(y)
      np.testing.assert_allclose(x_dx, event, rtol=RTOL)

  @parameterized.named_parameters(
      ('identity', lambda: Lambda(lambda x: x, is_constant_jacobian=True),
       tfb.Identity, np.array([1, 1.5, 2], dtype=np.float32)),
      ('affine', lambda: ScalarAffine(np.ones(3, dtype=np.float32),  # pylint: disable=g-long-lambda
                                      np.full(3, 5.5, dtype=np.float32)),
       lambda: tfb.Chain([tfb.Shift(np.ones(3, dtype=np.float32)),  # pylint: disable=g-long-lambda
                          tfb.Scale(np.full(3, 5.5, dtype=np.float32))]),
       np.array([1, 1.5, 2], dtype=np.float32)),
      ('tanh', Tanh, tfb.Tanh, np.array([-0.1, 0.01, 0.1], dtype=np.float32)),
      ('chain(tanh, affine)', lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]),
       np.array([-0.1, 0.01, 0.1], dtype=np.float32))
  )
  def test_forward_and_inverse(
      self, dx_bijector_fn, tfp_bijector_fn, event):
    dx_bij = tfp_compatible_bijector(dx_bijector_fn())
    tfp_bij = tfp_bijector_fn()

    with self.subTest('forward'):
      dx_out = dx_bij.forward(event)
      tfp_out = tfp_bij.forward(event)
      np.testing.assert_allclose(dx_out, tfp_out, rtol=RTOL)

    with self.subTest('inverse'):
      y = tfp_bij.forward(event)
      dx_out = dx_bij.inverse(y)
      tfp_out = tfp_bij.inverse(y)
      np.testing.assert_allclose(dx_out, tfp_out, rtol=RTOL)

  @parameterized.named_parameters(
      ('identity', lambda: Lambda(lambda x: x, is_constant_jacobian=True),
       tfb.Identity, np.array([1, 1.5, 2], dtype=np.float32)),
      ('affine', lambda: ScalarAffine(np.ones(3, dtype=np.float32),  # pylint: disable=g-long-lambda
                                      np.full(3, 5.5, dtype=np.float32)),
       lambda: tfb.Chain([tfb.Shift(np.ones(3, dtype=np.float32)),  # pylint: disable=g-long-lambda
                          tfb.Scale(np.full(3, 5.5, dtype=np.float32))]),
       np.array([1, 1.5, 2], dtype=np.float32)),
      ('tanh', Tanh, tfb.Tanh, np.array([-0.1, 0.01, 0.1], dtype=np.float32)),
      ('chain(tanh, affine)', lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]),
       lambda: tfb.Chain([tfb.Tanh(), tfb.Shift(0.2), tfb.Scale(1.2)]),
       np.array([-0.1, 0.01, 0.1], dtype=np.float32))
  )
  def test_log_det_jacobian(self, dx_bijector_fn, tfp_bijector_fn, event):
    base_bij = dx_bijector_fn()
    dx_bij = tfp_compatible_bijector(base_bij)
    tfp_bij = tfp_bijector_fn()

    with self.subTest('forward'):
      dx_out = dx_bij.forward_log_det_jacobian(
          event, event_ndims=base_bij.event_ndims_in)
      tfp_out = tfp_bij.forward_log_det_jacobian(
          event, event_ndims=base_bij.event_ndims_in)
      np.testing.assert_allclose(dx_out, tfp_out, rtol=RTOL)

    with self.subTest('inverse'):
      y = tfp_bij.forward(event)
      dx_out = dx_bij.inverse_log_det_jacobian(
          y, event_ndims=base_bij.event_ndims_out)
      tfp_out = tfp_bij.inverse_log_det_jacobian(
          y, event_ndims=base_bij.event_ndims_out)
      np.testing.assert_allclose(dx_out, tfp_out, rtol=RTOL)

    with self.subTest('experimental_compute_density_correction'):
      dx_out = dx_bij.forward_log_det_jacobian(
          event, event_ndims=base_bij.event_ndims_in)
      dx_dcorr_out, space = dx_bij.experimental_compute_density_correction(
          event, tangent_space=tfp.experimental.tangent_spaces.FullSpace(),
          event_ndims=base_bij.event_ndims_in)
      np.testing.assert_allclose(dx_out, dx_dcorr_out, rtol=RTOL)
      self.assertIsInstance(space, tfp.experimental.tangent_spaces.FullSpace)

  @parameterized.named_parameters(
      ('identity unbatched',
       lambda: Lambda(lambda x: x, is_constant_jacobian=True), ()),
      ('identity 1d-batch',
       lambda: Lambda(lambda x: x, is_constant_jacobian=True), (3,)),
      ('identity 2d-batch',
       lambda: Lambda(lambda x: x, is_constant_jacobian=True), (5, 3)),
      ('affine unbatched', lambda: ScalarAffine(1.0, 5.5), ()),
      ('affine 1d-batch', lambda: ScalarAffine(1.0, 5.5), (3,)),
      ('affine 2d-batch', lambda: ScalarAffine(1.0, 5.5), (5, 3)),
      ('tanh unbatched', Tanh, ()),
      ('tanh 1d-batch', Tanh, (3,)),
      ('tanh 2d-batch', Tanh, (5, 3)),
      ('chain(tanh, affine) unbatched',
       lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]), ()),
      ('chain(tanh, affine) 1d-batch',
       lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]), (3,)),
      ('chain(tanh, affine) 2d-batch',
       lambda: Chain([Tanh(), ScalarAffine(0.2, 1.2)]), (5, 3)),
  )
  def test_batched_events(self, bij_fn, batch_shape):
    base = tfd.MultivariateNormalDiag(np.zeros(batch_shape + (3,)),
                                      np.ones(batch_shape + (3,)))
    bij = tfp_compatible_bijector(bij_fn())
    dist = tfd.TransformedDistribution(base, bij)

    with self.subTest('sample'):
      sample = dist.sample(seed=jax.random.PRNGKey(0))
      chex.assert_shape(sample, batch_shape + (3,))

    with self.subTest('log_prob'):
      sample = dist.sample(seed=jax.random.PRNGKey(0))
      log_prob = dist.log_prob(sample)
      chex.assert_shape(log_prob, batch_shape)

  def test_with_different_event_ndims(self):
    dx_bij = Lambda(forward=lambda x: x.reshape(x.shape[:-1] + (2, 3)),
                    inverse=lambda y: y.reshape(y.shape[:-2] + (6,)),
                    forward_log_det_jacobian=lambda _: 0,
                    inverse_log_det_jacobian=lambda _: 0,
                    is_constant_jacobian=True,
                    event_ndims_in=1, event_ndims_out=2)
    tfp_bij = tfp_compatible_bijector(dx_bij)

    with self.subTest('forward_event_ndims'):
      assert tfp_bij.forward_event_ndims(1) == 2
      assert tfp_bij.forward_event_ndims(2) == 3

    with self.subTest('inverse_event_ndims'):
      assert tfp_bij.inverse_event_ndims(2) == 1
      assert tfp_bij.inverse_event_ndims(3) == 2

    with self.subTest('forward_event_ndims with incorrect input'):
      with self.assertRaises(ValueError):
        tfp_bij.forward_event_ndims(0)

    with self.subTest('inverse_event_ndims with incorrect input'):
      with self.assertRaises(ValueError):
        tfp_bij.inverse_event_ndims(0)

      with self.assertRaises(ValueError):
        tfp_bij.inverse_event_ndims(1)

    with self.subTest('forward_event_shape'):
      y_shape = tfp_bij.forward_event_shape((6,))
      y_shape_tensor = tfp_bij.forward_event_shape_tensor((6,))
      self.assertEqual(y_shape, (2, 3))
      np.testing.assert_array_equal(y_shape_tensor, jnp.array((2, 3)))

    with self.subTest('inverse_event_shape'):
      x_shape = tfp_bij.inverse_event_shape((2, 3))
      x_shape_tensor = tfp_bij.inverse_event_shape_tensor((2, 3))
      self.assertEqual(x_shape, (6,))
      np.testing.assert_array_equal(x_shape_tensor, jnp.array((6,)))

    with self.subTest('TransformedDistribution with correct event_ndims'):
      base = tfd.MultivariateNormalDiag(np.zeros(6), np.ones(6))
      dist = tfd.TransformedDistribution(base, tfp_bij)
      chex.assert_equal(dist.event_shape, (2, 3))

      sample = dist.sample(seed=jax.random.PRNGKey(0))
      chex.assert_shape(sample, (2, 3))

      log_prob = dist.log_prob(sample)
      chex.assert_shape(log_prob, ())

    with self.subTest('TransformedDistribution with incorrect event_ndims'):
      base = tfd.Normal(np.zeros(6), np.ones(6))
      dist = tfd.TransformedDistribution(base, tfp_bij)
      with self.assertRaises(ValueError):
        _ = dist.event_shape

if __name__ == '__main__':
  absltest.main()

