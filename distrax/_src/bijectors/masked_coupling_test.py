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
"""Tests for `masked_coupling.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import bijector as distrax_bijector
from distrax._src.bijectors import block
from distrax._src.bijectors import masked_coupling
import jax
import jax.numpy as jnp
import numpy as np


def _create_masked_coupling_bijector(event_shape, event_ndims=None):
  key = jax.random.PRNGKey(101)
  return masked_coupling.MaskedCoupling(
      mask=jax.random.choice(key, jnp.array([True, False]), event_shape),
      conditioner=lambda x: x**2,
      bijector=lambda _: lambda x: 2. * x + 3.,
      event_ndims=event_ndims)


class MaskedCouplingTest(parameterized.TestCase):

  def test_properties(self):
    bijector = _create_masked_coupling_bijector((4, 5), None)
    ones = jnp.ones((4, 5))
    np.testing.assert_allclose(bijector.conditioner(2 * ones), 4 * ones)
    assert callable(bijector.bijector(ones))
    self.assertEqual(bijector.mask.shape, (4, 5))

  @parameterized.named_parameters(
      ('jnp.float32', jnp.float32),
      ('jnp.int32', jnp.int32),
      ('jnp.float64', jnp.float64),
      ('jnp.int64', jnp.int64),
      ('jnp.complex64', jnp.complex64),
      ('jnp.complex128', jnp.complex128),
  )
  def test_raises_on_invalid_mask_dtype(self, dtype):
    with self.assertRaises(ValueError):
      masked_coupling.MaskedCoupling(
          mask=jnp.zeros((4,), dtype=dtype),
          conditioner=lambda x: x,
          bijector=lambda _: lambda x: x
      )

  @chex.all_variants
  @parameterized.parameters(
      {'event_ndims': None, 'batch_shape': (2, 3)},
      {'event_ndims': 0, 'batch_shape': (2, 3, 4, 5)},
      {'event_ndims': 1, 'batch_shape': (2, 3, 4)},
      {'event_ndims': 2, 'batch_shape': (2, 3)},
      {'event_ndims': 3, 'batch_shape': (2,)},
      {'event_ndims': 4, 'batch_shape': ()},
  )
  def test_shapes_are_correct(self, event_ndims, batch_shape):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _create_masked_coupling_bijector((4, 5), event_ndims)
    # Forward methods.
    y, logdet = self.variant(bijector.forward_and_log_det)(x)
    self.assertEqual(y.shape, (2, 3, 4, 5))
    self.assertEqual(logdet.shape, batch_shape)
    # Inverse methods.
    x, logdet = self.variant(bijector.inverse_and_log_det)(y)
    self.assertEqual(x.shape, (2, 3, 4, 5))
    self.assertEqual(logdet.shape, batch_shape)

  def test_non_default_inner_event_ndims(self):
    batch = 2
    event_shape = (7, 5, 3)
    inner_event_ndims = 1
    multipliers = jnp.array([4., 1., 0.5])

    class InnerBijector(distrax_bijector.Bijector):
      """A simple inner bijector."""

      def __init__(self):
        super().__init__(event_ndims_in=inner_event_ndims)

      def forward_and_log_det(self, x):
        return x * multipliers, jnp.full(x.shape[:-inner_event_ndims],
                                         jnp.sum(jnp.log(multipliers)))

      def inverse_and_log_det(self, y):
        return y / multipliers, jnp.full(x.shape[:-inner_event_ndims],
                                         -jnp.sum(jnp.log(multipliers)))

    bijector = masked_coupling.MaskedCoupling(
        mask=jnp.full(event_shape[:-inner_event_ndims], False),
        conditioner=lambda x: x,
        bijector=lambda _: InnerBijector(),
        inner_event_ndims=inner_event_ndims,
        event_ndims=len(event_shape))
    x = jnp.ones((batch,) + event_shape)
    # Test forward.
    y, ldj_y = bijector.forward_and_log_det(x)
    np.testing.assert_allclose(
        y, jnp.tile(multipliers[None, None, None, :], [batch, 7, 5, 1]))
    np.testing.assert_allclose(
        ldj_y,
        np.full([batch],
                np.prod(event_shape[:-inner_event_ndims]) *
                jnp.sum(jnp.log(multipliers))),
        rtol=1e-6)
    # Test inverse
    z, ldj_z = bijector.inverse_and_log_det(y)
    np.testing.assert_allclose(z, x)
    np.testing.assert_allclose(ldj_z, -ldj_y)

  @chex.all_variants
  def test_masking_works(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _create_masked_coupling_bijector((4, 5))
    mask = bijector.mask
    y = self.variant(bijector.forward)(x)
    np.testing.assert_array_equal(mask * y, mask * x)

  @chex.all_variants
  @parameterized.parameters(
      {'event_ndims': None},
      {'event_ndims': 0},
      {'event_ndims': 1},
      {'event_ndims': 2},
      {'event_ndims': 3},
      {'event_ndims': 4},
  )
  def test_inverse_methods_are_correct(self, event_ndims):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _create_masked_coupling_bijector((4, 5), event_ndims)
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    x_rec, logdet_inv = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_rec, x, atol=1e-6)
    np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)

  @chex.all_variants
  @parameterized.parameters(
      {'event_ndims': None},
      {'event_ndims': 0},
      {'event_ndims': 1},
      {'event_ndims': 2},
      {'event_ndims': 3},
      {'event_ndims': 4},
  )
  def test_composite_methods_are_consistent(self, event_ndims):
    key = jax.random.PRNGKey(42)
    bijector = _create_masked_coupling_bijector((4, 5), event_ndims)
    # Forward methods.
    x = jax.random.normal(key, (2, 3, 4, 5))
    y1 = self.variant(bijector.forward)(x)
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y1, y2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=5e-6)
    # Inverse methods.
    y = jax.random.normal(key, (2, 3, 4, 5))
    x1 = self.variant(bijector.inverse)(y)
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x1, x2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=5e-6)

  def test_raises_if_inner_bijector_is_not_scalar(self):
    key = jax.random.PRNGKey(101)
    event_shape = (2, 3)
    bijector = masked_coupling.MaskedCoupling(
        mask=jax.random.choice(key, jnp.array([True, False]), event_shape),
        conditioner=lambda x: x,
        bijector=lambda _: block.Block(lambda x: x, 1))
    with self.assertRaisesRegex(ValueError, r'match.*inner_event_ndims'):
      bijector.forward_and_log_det(jnp.zeros(event_shape))
    with self.assertRaisesRegex(ValueError, r'match.*inner_event_ndims'):
      bijector.inverse_and_log_det(jnp.zeros(event_shape))

  def test_raises_if_inner_bijector_has_wrong_inner_ndims(self):
    key = jax.random.PRNGKey(101)
    event_shape = (2, 3, 5)
    inner_event_ndims = 2
    bijector = masked_coupling.MaskedCoupling(
        mask=jax.random.choice(key, jnp.array([True, False]),
                               event_shape[:-inner_event_ndims]),
        event_ndims=len(event_shape),
        inner_event_ndims=inner_event_ndims,
        conditioner=lambda x: x,
        bijector=lambda _: block.Block(lambda x: x, 0))
    with self.assertRaisesRegex(ValueError, r'match.*inner_event_ndims'):
      bijector.forward_and_log_det(jnp.zeros(event_shape))
    with self.assertRaisesRegex(ValueError, r'match.*inner_event_ndims'):
      bijector.inverse_and_log_det(jnp.zeros(event_shape))

  def test_raises_on_invalid_input_shape(self):
    bij = _create_masked_coupling_bijector(event_shape=(2, 3))
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.assertRaises(ValueError):
        fn(jnp.zeros((3,)))

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
  def test_batched_mask(self, mask_batch_shape, input_batch_shape):
    def create_bijector(mask):
      return masked_coupling.MaskedCoupling(
          mask=mask,
          conditioner=lambda x: x**2,
          bijector=lambda _: lambda x: 2. * x + 3.,
          event_ndims=2)

    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    mask = jax.random.choice(
        k1, jnp.array([True, False]), mask_batch_shape + (5, 6))
    bijector = create_bijector(mask)

    x = jax.random.uniform(k2, input_batch_shape + (5, 6))
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    z, logdet_inv = self.variant(bijector.inverse_and_log_det)(x)

    output_batch_shape = jnp.broadcast_arrays(
        mask[..., 0, 0], x[..., 0, 0])[0].shape

    self.assertEqual(y.shape, output_batch_shape + (5, 6))
    self.assertEqual(z.shape, output_batch_shape + (5, 6))
    self.assertEqual(logdet_fwd.shape, output_batch_shape)
    self.assertEqual(logdet_inv.shape, output_batch_shape)

    mask = jnp.broadcast_to(
        mask, output_batch_shape + (5, 6)).reshape((-1, 5, 6))
    x = jnp.broadcast_to(x, output_batch_shape + (5, 6)).reshape((-1, 5, 6))
    y = y.reshape((-1, 5, 6))
    z = z.reshape((-1, 5, 6))
    logdet_fwd = logdet_fwd.flatten()
    logdet_inv = logdet_inv.flatten()

    for i in range(np.prod(output_batch_shape)):
      bijector = create_bijector(mask[i])
      this_y, this_logdet_fwd = self.variant(bijector.forward_and_log_det)(x[i])
      this_z, this_logdet_inv = self.variant(bijector.inverse_and_log_det)(x[i])
      np.testing.assert_allclose(this_y, y[i], atol=1e-7)
      np.testing.assert_allclose(this_z, z[i], atol=1e-7)
      np.testing.assert_allclose(this_logdet_fwd, logdet_fwd[i], atol=1e-5)
      np.testing.assert_allclose(this_logdet_inv, logdet_inv[i], atol=1e-5)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = _create_masked_coupling_bijector((4, 5), event_ndims=None)
    x = np.zeros((2, 3, 4, 5))
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
