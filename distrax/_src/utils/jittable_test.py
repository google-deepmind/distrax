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
"""Tests for `jittable.py`."""

from absl.testing import absltest
from absl.testing import parameterized

from distrax._src.distributions import normal
from distrax._src.utils import jittable
import jax
import jax.numpy as jnp
import numpy as np


class DummyJittable(jittable.Jittable):

  def __init__(self, params):
    self.name = 'dummy'  # Non-JAX property, cannot be traced.
    self.data = {'params': params}  # Tree property, must be traced recursively.


class JittableTest(parameterized.TestCase):

  def test_jittable(self):
    @jax.jit
    def get_params(obj):
      return obj.data['params']

    obj = DummyJittable(jnp.ones((5,)))
    np.testing.assert_array_equal(get_params(obj), obj.data['params'])

  def test_tree_map_preserves_structure_when_children_change_type(self):
    obj = DummyJittable(jnp.arange(3.0))
    mask = jax.tree_util.tree_map(lambda _: True, obj)

    self.assertEqual(
        jax.tree_util.tree_structure(obj),
        jax.tree_util.tree_structure(mask),
    )
    self.assertEqual(mask.name, 'dummy')
    self.assertTrue(mask.data['params'])
    restored = jax.tree_util.tree_map(
        lambda value, selected: value if selected else 0, obj, mask
    )
    np.testing.assert_array_equal(restored.data['params'], obj.data['params'])

  def test_adding_a_field_after_mapping_updates_dynamic_children(self):
    original = DummyJittable(jnp.array([1.0]))
    mapped = jax.tree_util.tree_map(lambda value: value + 1.0, original)
    mapped.extra = jnp.array([3.0])
    leaves = jax.tree_util.tree_leaves(mapped)

    self.assertLen(leaves, 2)
    np.testing.assert_array_equal(leaves[0], jnp.array([2.0]))
    np.testing.assert_array_equal(leaves[1], jnp.array([3.0]))

  def test_tree_map_preserves_structure_for_normal_distribution(self):
    distribution = normal.Normal(jnp.array(0.0), jnp.array(1.0))
    mask = jax.tree_util.tree_map(lambda _: True, distribution)
    self.assertEqual(
        jax.tree_util.tree_structure(distribution),
        jax.tree_util.tree_structure(mask),
    )
    result = jax.tree_util.tree_map(
        lambda x, selected: x if selected else 0, distribution, mask
    )
    np.testing.assert_array_equal(result.loc, distribution.loc)
    np.testing.assert_array_equal(result.scale, distribution.scale)

  def test_vmappable(self):
    def do_sum(obj):
      return obj.data['params'].sum()

    obj = DummyJittable(jnp.array([[1, 2, 3], [4, 5, 6]]))

    with self.subTest('no vmap'):
      np.testing.assert_array_equal(do_sum(obj), obj.data['params'].sum())

    with self.subTest('in_axes=0'):
      np.testing.assert_array_equal(
          jax.vmap(do_sum, in_axes=0)(obj), obj.data['params'].sum(axis=1)
      )

    with self.subTest('in_axes=1'):
      np.testing.assert_array_equal(
          jax.vmap(do_sum, in_axes=1)(obj), obj.data['params'].sum(axis=0)
      )

  def test_traceable(self):
    @jax.jit
    def inner_fn(obj):
      obj.data['params'] *= 3  # Modification after passing to jitted fn.
      return obj.data['params'].sum()

    def loss_fn(params):
      obj = DummyJittable(params)
      obj.data['params'] *= 2  # Modification before passing to jitted fn.
      return inner_fn(obj)

    with self.subTest('numpy'):
      params = np.ones((5,))
      # Both modifications will be traced if data tree is correctly traversed.
      grad_expected = params * 2 * 3
      grad = jax.grad(loss_fn)(params)
      np.testing.assert_array_equal(grad, grad_expected)

    with self.subTest('jax.numpy'):
      params = jnp.ones((5,))
      # Both modifications will be traced if data tree is correctly traversed.
      grad_expected = params * 2 * 3
      grad = jax.grad(loss_fn)(params)
      np.testing.assert_array_equal(grad, grad_expected)

  def test_different_jittables_to_compiled_function(self):
    @jax.jit
    def add_one_to_params(obj):
      obj.data['params'] = obj.data['params'] + 1
      return obj

    with self.subTest('numpy'):
      add_one_to_params(DummyJittable(np.zeros((5,))))
      add_one_to_params(DummyJittable(np.ones((5,))))

    with self.subTest('jax.numpy'):
      add_one_to_params(DummyJittable(jnp.zeros((5,))))
      add_one_to_params(DummyJittable(jnp.ones((5,))))

  def test_modifying_object_data_does_not_leak_tracers(self):
    @jax.jit
    def add_one_to_params(obj):
      obj.data['params'] = obj.data['params'] + 1
      return obj

    dummy = DummyJittable(jnp.ones((5,)))
    dummy_out = add_one_to_params(dummy)
    dummy_out.data['params'] -= 1

  def test_metadata_modification_statements_are_removed_by_compilation(self):
    @jax.jit
    def add_char_to_name(obj):
      obj.name += '_x'
      return obj

    dummy = DummyJittable(jnp.ones((5,)))
    dummy_out = add_char_to_name(dummy)
    dummy_out = add_char_to_name(dummy)  # `name` change has been compiled out.
    dummy_out.name += 'y'
    self.assertEqual(dummy_out.name, 'dummy_xy')


if __name__ == '__main__':
  absltest.main()
