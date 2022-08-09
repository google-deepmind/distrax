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
"""Tests for `distribution.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import distribution
import jax
import jax.numpy as jnp
import numpy as np


class DummyUnivariateDist(distribution.Distribution):
  """Dummy univariate distribution for testing."""

  def _sample_n(self, key, n):
    return jax.random.uniform(key, shape=(n,))

  def log_prob(self, value):
    """Log probability density/mass function."""

  @property
  def event_shape(self):
    """Shape of the events."""
    return jnp.shape([])


class DummyMultivariateDist(distribution.Distribution):
  """Dummy multivariate distribution for testing."""

  def __init__(self, dimension):
    super().__init__()
    self._dimension = dimension

  def _sample_n(self, key, n):
    return jax.random.uniform(key, shape=(n,) + self._dimension)

  def log_prob(self, value):
    """Log probability density/mass function."""

  @property
  def event_shape(self):
    """Shape of the events."""
    return (self._dimension,)


class DummyNestedDist(distribution.Distribution):
  """Dummy distribution with nested events for testing."""

  def __init__(self, batch_shape=()) -> None:
    self._batch_shape = batch_shape

  def _sample_n(self, key, n):
    return dict(
        foo=jax.random.uniform(key, shape=(n,) + self._batch_shape),
        bar=jax.random.uniform(key, shape=(n,) + self._batch_shape + (3,)))

  def log_prob(self, value):
    """Log probability density/mass function."""

  @property
  def event_shape(self):
    """Shape of the events."""
    return dict(foo=(), bar=(3,))


class DistributionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.uni_dist = DummyUnivariateDist()

  @chex.all_variants(with_jit=False, with_device=False, with_pmap=False)
  @parameterized.named_parameters(
      ('0d input', 1, (1,)),
      ('0d np.int16 input', np.int16(2), (2,)),
      ('0d np.int32 input', np.int32(2), (2,)),
      ('0d np.int64 input', np.int64(2), (2,)),
      ('1d tuple input', (2,), (2,)),
      ('1d list input', [2], (2,)),
      ('1d tuple of np.int32 input', (np.int32(2),), (2,)),
      ('2d input', (2, 3), (2, 3)),
      ('3d input', (2, 3, 4), (2, 3, 4)))
  def test_sample_univariate_shape(self, shape, expected_shape):
    sample_fn = self.variant(
        lambda key: self.uni_dist.sample(seed=key, sample_shape=shape))
    samples = sample_fn(0)
    np.testing.assert_equal(samples.shape, expected_shape)

  @chex.all_variants(with_jit=False, with_device=False, with_pmap=False)
  @parameterized.named_parameters(
      ('0d input', (5,), 1, (1, 5)),
      ('0d np.int16 input', (5,), np.int16(1), (1, 5)),
      ('0d np.int32 input', (5,), np.int32(1), (1, 5)),
      ('0d np.int64 input', (5,), np.int64(1), (1, 5)),
      ('1d tuple input', (5,), (2,), (2, 5)),
      ('1d list input', (5,), [2], (2, 5)),
      ('1d tuple of np.int32 input', (5,), (np.int32(2),), (2, 5)),
      ('2d input', (4, 5), (2, 3), (2, 3, 4, 5)))
  def test_sample_multivariate_shape(self, var_dim, shape, expected_shape):
    mult_dist = DummyMultivariateDist(var_dim)
    sample_fn = self.variant(
        lambda key: mult_dist.sample(seed=key, sample_shape=shape))
    samples = sample_fn(0)
    np.testing.assert_equal(samples.shape, expected_shape)

  @chex.all_variants(with_jit=False, with_device=False, with_pmap=False)
  @parameterized.named_parameters(
      ('0d input', 1),
      ('0d np.int16 input', np.int16(1)),
      ('0d np.int32 input', np.int32(1)),
      ('0d np.int64 input', np.int64(1)),
      ('1d tuple input', (2,)),
      ('1d list input', [2]),
      ('2d input', (2, 3)),
  )
  def test_sample_nested_shape(self, shape):
    dist = DummyNestedDist()
    sample_fn = self.variant(
        lambda key: dist.sample(seed=key, sample_shape=shape))
    samples = sample_fn(0)
    # Ensure shape is a tuple.
    try:
      iter(shape)
    except TypeError:
      shape = (shape,)
    shape = tuple(shape)
    np.testing.assert_equal(samples['foo'].shape, shape)
    np.testing.assert_equal(samples['bar'].shape, shape + (3,))

  @parameterized.named_parameters(
      ('empty batch', ()),
      ('1d batch', (3,)),
      ('2d batch', (3, 4)),
  )
  def test_nested_batch_shape(self, batch_shape):
    dist = DummyNestedDist(batch_shape=batch_shape)
    np.testing.assert_equal(dist.batch_shape, batch_shape)

  @chex.all_variants(with_jit=False, with_device=False, with_pmap=False)
  def test_sample_keys(self):
    shape = 5
    key = 0
    sample_fn = self.variant(
        lambda key: self.uni_dist.sample(seed=key, sample_shape=shape))
    samples_from_int = sample_fn(key)
    rng = jax.random.PRNGKey(key)
    samples_from_prngkey = sample_fn(rng)
    np.testing.assert_array_equal(samples_from_int, samples_from_prngkey)

  def test_jittable(self):

    @jax.jit
    def sampler(dist, seed):
      return dist.sample(seed=seed)

    seed = jax.random.PRNGKey(0)
    dist = DummyMultivariateDist((5,))
    np.testing.assert_array_equal(
        sampler(dist, seed=seed), dist.sample(seed=seed))

  @parameterized.named_parameters(
      ('int', int),
      ('np.int16', np.int16),
      ('np.int32', np.int32),
      ('np.int64', np.int64),
      ('PRNGKey', jax.random.PRNGKey),
  )
  def test_convert_seed(self, dtype):
    rng, _ = distribution.convert_seed_and_sample_shape(dtype(0), 2)
    jax.random.split(rng)  # Should not raise an error.

  @parameterized.named_parameters(
      ('int', 2, (2,)),
      ('np.int16', np.int16(2), (2,)),
      ('np.int32', np.int32(2), (2,)),
      ('np.int64', np.int64(2), (2,)),
      ('int tuple', (2, 3), (2, 3)),
      ('np.int16 tuple', (np.int16(2), np.int16(3)), (2, 3)),
      ('np.int32 tuple', (np.int32(2), np.int32(3)), (2, 3)),
      ('np.int64 tuple', (np.int64(2), np.int64(3)), (2, 3)),
      ('int list', [2, 3], (2, 3)),
      ('np.int16 list', [np.int16(2), np.int16(3)], (2, 3)),
      ('np.int32 list', [np.int32(2), np.int32(3)], (2, 3)),
      ('np.int64 list', [np.int64(2), np.int64(3)], (2, 3)),
  )
  def test_convert_sample_shape(self, shape_in, shape_out):
    _, sample_shape = distribution.convert_seed_and_sample_shape(0, shape_in)
    assert sample_shape == shape_out

  @parameterized.named_parameters(
      ('single', 0, (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                     np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]),
                     np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]))),
      ('range', slice(-1),
       (np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                 ]), np.array([[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]]),
        np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]))),
      ('two_axes', (slice(None), 1), (np.array(
          [[0, 0, 0, 0], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
                                      np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))),
      ('ellipsis', (Ellipsis, 2),
       (np.array([[0, 0, 0], [1, 1, 1]]), np.array(
           [[0, 1, 2], [0, 1, 2]]), np.array([[2, 2, 2], [2, 2, 2]]))),
      ('np_array', np.array([0, 1, -1]),
       ((np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                   [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                   [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
                  dtype=np.int32),
         np.array([[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                   [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                   [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]],
                  dtype=np.int32),
         np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                   [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                   [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]],
                  dtype=np.int32)))),
  )
  def test_to_batch_shape_index(self, index, expected):
    np.testing.assert_allclose(
        distribution.to_batch_shape_index(batch_shape=(2, 3, 4), index=index),
        expected, 1e-3)

  def test_to_batch_shape_index_jnp_array(self):
    # This test needs to be a separate function since JAX doesn't allow creating
    # jnp.arrays in the top level of the program.
    # NOTE: Using jnp.arrays behaves differently compared to np.arrays as it
    # wraps instead of raising. Raising for same index is tested for np.arrays
    # below.
    index = (-1, 0)
    expected = (jnp.array([1, 1, 1, 1], dtype=jnp.int32),
                jnp.array([0, 0, 0, 0], dtype=jnp.int32),
                jnp.array([0, 1, 2, 3], dtype=jnp.int32))
    np.testing.assert_allclose(
        distribution.to_batch_shape_index(batch_shape=(2, 3, 4), index=index),
        expected, 1e-3)

  @parameterized.named_parameters(
      ('long_index', (1, 2, 3, 4)),
      ('np_array_out_of_bounds', np.array([-1, 2])),
  )
  def test_to_batch_shape_index_raises(self, index):
    with self.assertRaisesRegex(IndexError, 'not compatible with index'):
      distribution.to_batch_shape_index(
          batch_shape=(2, 3, 4), index=index)

  def test_multivariate_survival_function_raises(self):
    mult_dist = DummyMultivariateDist(42)
    with self.assertRaises(NotImplementedError):
      mult_dist.survival_function(jnp.zeros(42))
    with self.assertRaises(NotImplementedError):
      mult_dist.log_survival_function(jnp.zeros(42))

if __name__ == '__main__':
  absltest.main()
