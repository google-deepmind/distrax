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
"""Tests for `split_coupling.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import bijector as base_bijector
from distrax._src.bijectors import block
from distrax._src.bijectors import split_coupling
import jax
import jax.numpy as jnp
import numpy as np


def _create_split_coupling_bijector(split_index,
                                    split_axis=-1,
                                    swap=False,
                                    event_ndims=2):
  return split_coupling.SplitCoupling(
      split_index=split_index,
      split_axis=split_axis,
      event_ndims=event_ndims,
      swap=swap,
      conditioner=lambda x: x**2,
      bijector=lambda _: lambda x: 2. * x + 3.)


class DummyBijector(base_bijector.Bijector):

  def forward_and_log_det(self, x):
    super()._check_forward_input_shape(x)
    return x, jnp.zeros((x.shape[:-self.event_ndims_in]), dtype=jnp.float_)


class SplitCouplingTest(parameterized.TestCase):

  def test_properties(self):
    bijector = _create_split_coupling_bijector(
        split_index=0, swap=False, split_axis=-1, event_ndims=2)
    ones = jnp.ones((4, 5))
    self.assertEqual(bijector.split_index, 0)
    self.assertEqual(bijector.split_axis, -1)
    self.assertFalse(bijector.swap)
    np.testing.assert_allclose(
        bijector.conditioner(2 * ones), 4 * ones, atol=1e-4)
    assert callable(bijector.bijector(ones))

  @parameterized.named_parameters(
      ('negative split_index', {'split_index': -1, 'event_ndims': 0}),
      ('positive split_axis',
       {'split_index': 0, 'event_ndims': 0, 'split_axis': 3}),
      ('negative event_ndims', {'split_index': 0, 'event_ndims': -1}),
      ('invalid split_axis',
       {'split_index': 0, 'event_ndims': 1, 'split_axis': -2}),
  )
  def test_invalid_properties(self, bij_params):
    bij_params.update(
        {'conditioner': lambda x: x, 'bijector': lambda _: lambda x: x})
    with self.assertRaises(ValueError):
      split_coupling.SplitCoupling(**bij_params)

  def test_raises_on_bijector_with_different_event_ndims(self):
    inner_bij = lambda _: DummyBijector(1, 0, False, False)
    bij_params = {'split_index': 0, 'event_ndims': 1,
                  'conditioner': lambda x: x, 'bijector': inner_bij}
    bij = split_coupling.SplitCoupling(**bij_params)
    with self.assertRaises(ValueError):
      bij.forward_and_log_det(jnp.zeros((4, 3)))

  def test_raises_on_bijector_with_extra_event_ndims(self):
    inner_bij = lambda _: DummyBijector(2, 2, False, False)
    bij_params = {'split_index': 0, 'event_ndims': 1,
                  'conditioner': lambda x: x, 'bijector': inner_bij}
    bij = split_coupling.SplitCoupling(**bij_params)
    with self.assertRaises(ValueError):
      bij.forward_and_log_det(jnp.zeros((4, 3)))

  @chex.all_variants
  @parameterized.parameters(
      {'split_index': 0, 'split_axis': -1, 'swap': False},
      {'split_index': 3, 'split_axis': -1, 'swap': False},
      {'split_index': 5, 'split_axis': -1, 'swap': False},
      {'split_index': 0, 'split_axis': -2, 'swap': False},
      {'split_index': 2, 'split_axis': -2, 'swap': False},
      {'split_index': 4, 'split_axis': -2, 'swap': False},
      {'split_index': 0, 'split_axis': -1, 'swap': True},
      {'split_index': 3, 'split_axis': -1, 'swap': True},
      {'split_index': 5, 'split_axis': -1, 'swap': True},
      {'split_index': 0, 'split_axis': -2, 'swap': True},
      {'split_index': 2, 'split_axis': -2, 'swap': True},
      {'split_index': 4, 'split_axis': -2, 'swap': True},
  )
  def test_shapes_are_correct(self, split_index, split_axis, swap):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _create_split_coupling_bijector(
        split_index, split_axis, swap, event_ndims=2)
    # Forward methods.
    y, logdet = self.variant(bijector.forward_and_log_det)(x)
    self.assertEqual(y.shape, (2, 3, 4, 5))
    self.assertEqual(logdet.shape, (2, 3))
    # Inverse methods.
    x, logdet = self.variant(bijector.inverse_and_log_det)(y)
    self.assertEqual(x.shape, (2, 3, 4, 5))
    self.assertEqual(logdet.shape, (2, 3))

  @chex.all_variants
  def test_swapping_works(self):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    # Don't swap.
    bijector = _create_split_coupling_bijector(
        split_index=3, split_axis=-1, swap=False)
    y = self.variant(bijector.forward)(x)
    np.testing.assert_array_equal(y[..., :3], x[..., :3])
    # Swap.
    bijector = _create_split_coupling_bijector(
        split_index=3, split_axis=-1, swap=True)
    y = self.variant(bijector.forward)(x)
    np.testing.assert_array_equal(y[..., 3:], x[..., 3:])
    # Don't swap.
    bijector = _create_split_coupling_bijector(
        split_index=3, split_axis=-2, swap=False)
    y = self.variant(bijector.forward)(x)
    np.testing.assert_array_equal(y[..., :3, :], x[..., :3, :])
    # Swap.
    bijector = _create_split_coupling_bijector(
        split_index=3, split_axis=-2, swap=True)
    y = self.variant(bijector.forward)(x)
    np.testing.assert_array_equal(y[..., 3:, :], x[..., 3:, :])

  @chex.all_variants
  @parameterized.parameters(
      {'split_index': 0, 'split_axis': -1, 'swap': False},
      {'split_index': 3, 'split_axis': -1, 'swap': False},
      {'split_index': 5, 'split_axis': -1, 'swap': False},
      {'split_index': 0, 'split_axis': -2, 'swap': False},
      {'split_index': 2, 'split_axis': -2, 'swap': False},
      {'split_index': 4, 'split_axis': -2, 'swap': False},
      {'split_index': 0, 'split_axis': -1, 'swap': True},
      {'split_index': 3, 'split_axis': -1, 'swap': True},
      {'split_index': 5, 'split_axis': -1, 'swap': True},
      {'split_index': 0, 'split_axis': -2, 'swap': True},
      {'split_index': 2, 'split_axis': -2, 'swap': True},
      {'split_index': 4, 'split_axis': -2, 'swap': True},
  )
  def test_inverse_methods_are_correct(self, split_index, split_axis, swap):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 3, 4, 5))
    bijector = _create_split_coupling_bijector(
        split_index, split_axis, swap, event_ndims=2)
    y, logdet_fwd = self.variant(bijector.forward_and_log_det)(x)
    x_rec, logdet_inv = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_rec, x, atol=1e-6)
    np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)

  @chex.all_variants
  @parameterized.parameters(
      {'split_index': 0, 'split_axis': -1, 'swap': False},
      {'split_index': 3, 'split_axis': -1, 'swap': False},
      {'split_index': 5, 'split_axis': -1, 'swap': False},
      {'split_index': 0, 'split_axis': -2, 'swap': False},
      {'split_index': 2, 'split_axis': -2, 'swap': False},
      {'split_index': 4, 'split_axis': -2, 'swap': False},
      {'split_index': 0, 'split_axis': -1, 'swap': True},
      {'split_index': 3, 'split_axis': -1, 'swap': True},
      {'split_index': 5, 'split_axis': -1, 'swap': True},
      {'split_index': 0, 'split_axis': -2, 'swap': True},
      {'split_index': 2, 'split_axis': -2, 'swap': True},
      {'split_index': 4, 'split_axis': -2, 'swap': True},
  )
  def test_composite_methods_are_consistent(self, split_index, split_axis,
                                            swap):
    key = jax.random.PRNGKey(42)
    bijector = _create_split_coupling_bijector(
        split_index, split_axis, swap, event_ndims=2)
    # Forward methods.
    x = jax.random.normal(key, (2, 3, 4, 5))
    y1 = self.variant(bijector.forward)(x)
    logdet1 = self.variant(bijector.forward_log_det_jacobian)(x)
    y2, logdet2 = self.variant(bijector.forward_and_log_det)(x)
    np.testing.assert_allclose(y1, y2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-8)
    # Inverse methods.
    y = jax.random.normal(key, (2, 3, 4, 5))
    x1 = self.variant(bijector.inverse)(y)
    logdet1 = self.variant(bijector.inverse_log_det_jacobian)(y)
    x2, logdet2 = self.variant(bijector.inverse_and_log_det)(y)
    np.testing.assert_allclose(x1, x2, atol=1e-8)
    np.testing.assert_allclose(logdet1, logdet2, atol=1e-8)

  def test_raises_on_invalid_input_shape(self):
    event_shape = (2, 3)
    bij = split_coupling.SplitCoupling(
        split_index=event_shape[-1] // 2,
        event_ndims=len(event_shape),
        conditioner=lambda x: x,
        bijector=lambda _: lambda x: x)
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.assertRaises(ValueError):
        fn(jnp.zeros((3,)))

  def test_raises_on_invalid_inner_bijector(self):
    event_shape = (2, 3)
    bij = split_coupling.SplitCoupling(
        split_index=event_shape[-1] // 2,
        event_ndims=len(event_shape),
        conditioner=lambda x: x,
        bijector=lambda _: block.Block(lambda x: x, len(event_shape) + 1))
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.assertRaises(ValueError):
        fn(jnp.zeros(event_shape))

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = _create_split_coupling_bijector(0, -1, False, event_ndims=2)
    x = np.zeros((2, 3, 4, 5))
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
