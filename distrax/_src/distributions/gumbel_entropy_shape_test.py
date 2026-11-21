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
"""Regression tests for broadcast Gumbel entropy."""

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np


class GumbelEntropyShapeTest(parameterized.TestCase):

  @parameterized.product(
      shapes=(
          ((), ()),
          ((2, 3), ()),
          ((2, 1), (3,)),
          ((), (2, 3)),
          ((0, 3), (3,)),
      ),
      compiled=(False, True),
  )
  def test_entropy_has_the_complete_batch_shape(self, shapes, compiled):
    loc_shape, scale_shape = shapes
    loc = jnp.zeros(loc_shape, dtype=jnp.float32)
    scale = jnp.full(scale_shape, 2.0, dtype=jnp.float32)
    dist = distrax.Gumbel(loc, scale)
    entropy_fn = lambda loc, scale: distrax.Gumbel(loc, scale).entropy()
    if compiled:
      entropy_fn = jax.jit(entropy_fn)
    result = entropy_fn(loc, scale)
    expected_shape = np.broadcast_shapes(loc_shape, scale_shape)
    self.assertEqual(dist.batch_shape, expected_shape)
    self.assertEqual(result.shape, expected_shape)
    self.assertEqual(result.dtype, jnp.float32)
    expected = np.full(expected_shape, np.log(2.0) + 1.0 + np.euler_gamma)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

  @parameterized.parameters(1, 2)
  def test_independent_entropy_sums_reinterpreted_dimensions(self, ndims):
    dist = distrax.Independent(distrax.Gumbel(jnp.zeros((2, 3)), 2.0), ndims)
    result = jax.jit(lambda: dist.entropy())()
    self.assertEqual(result.shape, dist.batch_shape)
    factor = 3 if ndims == 1 else 6
    np.testing.assert_allclose(
        result,
        np.full(
            dist.batch_shape, factor * (np.log(2.0) + 1.0 + np.euler_gamma)
        ),
        rtol=1e-6,
    )

  def test_entropy_gradients_count_all_batch_members(self):
    loc = jnp.zeros((2, 3))
    scale = jnp.array(2.0)
    _, (loc_grad, scale_grad) = jax.jit(
        jax.value_and_grad(
            lambda loc, scale: jnp.sum(distrax.Gumbel(loc, scale).entropy()),
            argnums=(0, 1),
        )
    )(loc, scale)
    np.testing.assert_array_equal(loc_grad, np.zeros((2, 3)))
    np.testing.assert_allclose(scale_grad, 6 / 2.0, rtol=1e-6)

  def test_vmap_and_slicing_keep_entropy_shapes(self):
    loc = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
    dist = distrax.Gumbel(loc, 2.0)
    expected = np.full((2, 3), np.log(2.0) + 1.0 + np.euler_gamma)
    result = jax.vmap(lambda loc: distrax.Gumbel(loc, 2.0).entropy())(loc)
    self.assertEqual(result.shape, (2, 3))
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    np.testing.assert_allclose(dist[0].entropy(), expected[0], rtol=1e-6)
    self.assertEqual(dist[0].entropy().shape, (3,))


if __name__ == '__main__':
  absltest.main()
