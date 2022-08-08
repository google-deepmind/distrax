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
"""Tests for `block.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import bijector as base
from distrax._src.bijectors import block as block_bijector
from distrax._src.bijectors import scalar_affine
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


RTOL = 1e-6


class BlockTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  def test_properties(self):
    bijct = conversion.as_bijector(jnp.tanh)
    block = block_bijector.Block(bijct, 1)
    assert block.ndims == 1
    assert isinstance(block.bijector, base.Bijector)

  def test_invalid_properties(self):
    bijct = conversion.as_bijector(jnp.tanh)
    with self.assertRaises(ValueError):
      block_bijector.Block(bijct, -1)

  @chex.all_variants
  @parameterized.named_parameters(
      ('scale_0', lambda: tfb.Scale(2), 0),
      ('scale_1', lambda: tfb.Scale(2), 1),
      ('scale_2', lambda: tfb.Scale(2), 2),
      ('reshape_0', lambda: tfb.Reshape([120], [4, 5, 6]), 0),
      ('reshape_1', lambda: tfb.Reshape([120], [4, 5, 6]), 1),
      ('reshape_2', lambda: tfb.Reshape([120], [4, 5, 6]), 2),
  )
  def test_against_tfp_semantics(self, tfp_bijector_fn, ndims):
    tfp_bijector = tfp_bijector_fn()
    x = jax.random.normal(self.seed, [2, 3, 4, 5, 6])
    y = tfp_bijector(x)
    fwd_event_ndims = ndims + tfp_bijector.forward_min_event_ndims
    inv_event_ndims = ndims + tfp_bijector.inverse_min_event_ndims
    block = block_bijector.Block(tfp_bijector, ndims)
    np.testing.assert_allclose(
        tfp_bijector.forward_log_det_jacobian(x, fwd_event_ndims),
        self.variant(block.forward_log_det_jacobian)(x), atol=2e-5)
    np.testing.assert_allclose(
        tfp_bijector.inverse_log_det_jacobian(y, inv_event_ndims),
        self.variant(block.inverse_log_det_jacobian)(y), atol=2e-5)

  @chex.all_variants
  @parameterized.named_parameters(
      ('dx_tanh_0', lambda: jnp.tanh, 0),
      ('dx_tanh_1', lambda: jnp.tanh, 1),
      ('dx_tanh_2', lambda: jnp.tanh, 2),
      ('tfp_tanh_0', tfb.Tanh, 0),
      ('tfp_tanh_1', tfb.Tanh, 1),
      ('tfp_tanh_2', tfb.Tanh, 2),
  )
  def test_forward_inverse_work_as_expected(self, bijector_fn, ndims):
    bijct = conversion.as_bijector(bijector_fn())
    x = jax.random.normal(self.seed, [2, 3])
    block = block_bijector.Block(bijct, ndims)
    np.testing.assert_array_equal(
        self.variant(bijct.forward)(x),
        self.variant(block.forward)(x))
    np.testing.assert_array_equal(
        self.variant(bijct.inverse)(x),
        self.variant(block.inverse)(x))
    np.testing.assert_allclose(
        self.variant(bijct.forward_and_log_det)(x)[0],
        self.variant(block.forward_and_log_det)(x)[0], atol=2e-7)
    np.testing.assert_array_equal(
        self.variant(bijct.inverse_and_log_det)(x)[0],
        self.variant(block.inverse_and_log_det)(x)[0])

  @chex.all_variants
  @parameterized.named_parameters(
      ('dx_tanh_0', lambda: jnp.tanh, 0),
      ('dx_tanh_1', lambda: jnp.tanh, 1),
      ('dx_tanh_2', lambda: jnp.tanh, 2),
      ('tfp_tanh_0', tfb.Tanh, 0),
      ('tfp_tanh_1', tfb.Tanh, 1),
      ('tfp_tanh_2', tfb.Tanh, 2),
  )
  def test_log_det_jacobian_works_as_expected(self, bijector_fn, ndims):
    bijct = conversion.as_bijector(bijector_fn())
    x = jax.random.normal(self.seed, [2, 3])
    block = block_bijector.Block(bijct, ndims)
    axes = tuple(range(-ndims, 0))
    np.testing.assert_allclose(
        self.variant(bijct.forward_log_det_jacobian)(x).sum(axes),
        self.variant(block.forward_log_det_jacobian)(x), rtol=RTOL)
    np.testing.assert_allclose(
        self.variant(bijct.inverse_log_det_jacobian)(x).sum(axes),
        self.variant(block.inverse_log_det_jacobian)(x), rtol=RTOL)
    np.testing.assert_allclose(
        self.variant(bijct.forward_and_log_det)(x)[1].sum(axes),
        self.variant(block.forward_and_log_det)(x)[1], rtol=RTOL)
    np.testing.assert_allclose(
        self.variant(bijct.inverse_and_log_det)(x)[1].sum(axes),
        self.variant(block.inverse_and_log_det)(x)[1], rtol=RTOL)

  def test_raises_on_invalid_input_shape(self):
    bij = block_bijector.Block(lambda x: x, 1)
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.assertRaises(ValueError):
        fn(jnp.array(0))

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = block_bijector.Block(scalar_affine.ScalarAffine(0), 1)
    x = np.zeros((2, 3))
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
