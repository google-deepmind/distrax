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
"""Tests for `mvn_full_covariance.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


def _sample_covariance_matrix(rng, shape):
  matrix = rng.normal(size=shape)
  matrix_t = np.vectorize(np.transpose, signature='(k,k)->(k,k)')(matrix)
  return np.matmul(matrix, matrix_t)


class MultivariateNormalFullCovarianceTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(MultivariateNormalFullCovariance)

  @parameterized.named_parameters(
      ('all inputs are None', {}),
      ('wrong dimension of loc', {
          'loc': np.array(0.),
      }),
      ('covariance_matrix is 0d', {
          'covariance_matrix': np.array(1.),
      }),
      ('covariance_matrix is 1d', {
          'covariance_matrix': np.ones((4,)),
      }),
      ('covariance_matrix is not square', {
          'covariance_matrix': np.ones((4, 3)),
      }),
      ('inconsistent loc and covariance_matrix', {
          'loc': np.zeros((4,)),
          'covariance_matrix': np.eye(5),
      }),
  )
  def test_raises_on_wrong_inputs(self, dist_kwargs):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_kwargs)

  @parameterized.named_parameters(
      ('loc provided', {'loc': np.zeros((4,))}),
      ('covariance_matrix provided', {'covariance_matrix': np.eye(4)}),
  )
  def test_default_properties(self, dist_kwargs):
    dist = self.distrax_cls(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.loc, jnp.zeros((4,)))
    self.assertion_fn(rtol=1e-3)(dist.covariance_matrix, jnp.eye(4))

  @parameterized.named_parameters(
      ('unbatched', (), (4,), (4, 4)),
      ('batched loc', (7,), (7, 4), (4, 4)),
      ('batched covariance_matrix', (7,), (4,), (7, 4, 4)),
  )
  def test_properties(self, batch_shape, loc_shape, covariance_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    covariance_matrix = _sample_covariance_matrix(rng, covariance_matrix_shape)
    dist = self.distrax_cls(loc=loc, covariance_matrix=covariance_matrix)
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=1e-3)(
        dist.loc, jnp.broadcast_to(loc, batch_shape + (4,)))
    self.assertion_fn(rtol=1e-3)(dist.covariance_matrix, jnp.broadcast_to(
        covariance_matrix, batch_shape + (4, 4)))

  @chex.all_variants
  @parameterized.named_parameters(
      ('unbatched, no shape', (), (4,), (4, 4)),
      ('batched loc, no shape', (), (7, 4), (4, 4)),
      ('batched covariance_matrix, no shape', (), (4,), (7, 4, 4)),
      ('unbatched, with shape', (3,), (4,), (4, 4)),
      ('batched loc, with shape', (3,), (7, 4), (4, 4)),
      ('batched covariance_matrix, with shape', (3,), (4,), (7, 4, 4)),
  )
  def test_sample_shape(self, sample_shape, loc_shape, covariance_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    covariance_matrix = _sample_covariance_matrix(rng, covariance_matrix_shape)
    dist_kwargs = {'loc': loc, 'covariance_matrix': covariance_matrix}
    super()._test_sample_shape(
        dist_args=(), dist_kwargs=dist_kwargs, sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist_params = {
        'loc': np.array([0., 0.], dtype),
        'covariance_matrix': np.array([[1., 0.], [0., 1.]], dtype)}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('unbatched, unbatched value', (4,), (4,), (4, 4)),
      ('batched loc, unbatched value', (4,), (7, 4), (4, 4)),
      ('batched covariance_matrix, unbatched value', (4,), (4,), (7, 4, 4)),
      ('unbatched, batched value', (3, 7, 4), (4,), (4, 4)),
      ('batched loc, batched value', (3, 7, 4), (7, 4), (4, 4)),
      ('batched covariance_matrix, batched value', (3, 7, 4), (4,), (7, 4, 4)),
  )
  def test_log_prob(self, value_shape, loc_shape, covariance_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    covariance_matrix = _sample_covariance_matrix(rng, covariance_matrix_shape)
    dist_kwargs = {'loc': loc, 'covariance_matrix': covariance_matrix}
    value = rng.normal(size=value_shape)
    super()._test_attribute(
        attribute_string='log_prob',
        dist_kwargs=dist_kwargs,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('unbatched', (4,), (4, 4)),
      ('batched loc', (7, 4), (4, 4)),
      ('batched covariance_matrix', (4,), (7, 4, 4)),
  )
  def test_method(self, loc_shape, covariance_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    covariance_matrix = _sample_covariance_matrix(rng, covariance_matrix_shape)
    dist_kwargs = {'loc': loc, 'covariance_matrix': covariance_matrix}
    for method in ['entropy', 'mean', 'stddev', 'variance',
                   'covariance', 'mode']:
      if method == 'covariance':
        rtol = 2e-2
      elif method in ['stddev', 'variance']:
        rtol = 6e-3
      else:
        rtol = 1e-3
      with self.subTest(method=method):
        super()._test_attribute(
            method,
            dist_kwargs=dist_kwargs,
            assertion_fn=self.assertion_fn(rtol=rtol))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_two_distributions(self, function_string, mode_string):
    rng = np.random.default_rng(2022)
    dist1_kwargs = {
        'loc': rng.normal(size=(5, 1, 4)),
        'covariance_matrix': _sample_covariance_matrix(rng, (3, 4, 4)),
    }
    dist2_kwargs = {
        'loc': rng.normal(size=(3, 4)),
        'covariance_matrix': _sample_covariance_matrix(rng, (4, 4)),
    }
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs=dist1_kwargs,
        dist2_kwargs=dist2_kwargs,
        assertion_fn=self.assertion_fn(rtol=1e-3))

  def test_jittable(self):
    super()._test_jittable(
        dist_kwargs={'loc': np.zeros((4,))},
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=(6, 5, 4))
    covariance_matrix = _sample_covariance_matrix(rng, (4, 4))
    dist_kwargs = {'loc': loc, 'covariance_matrix': covariance_matrix}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertEqual(dist[slice_].batch_shape, loc[slice_].shape[:-1])
    self.assertEqual(dist[slice_].event_shape, dist.event_shape)
    self.assertion_fn(rtol=1e-3)(dist[slice_].mean(), loc[slice_])

  def test_slice_ellipsis(self):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=(6, 5, 4))
    covariance_matrix = _sample_covariance_matrix(rng, (4, 4))
    dist_kwargs = {'loc': loc, 'covariance_matrix': covariance_matrix}
    dist = self.distrax_cls(**dist_kwargs)
    self.assertEqual(dist[..., -1].batch_shape, (6,))
    self.assertEqual(dist[..., -1].event_shape, dist.event_shape)
    self.assertion_fn(rtol=1e-3)(dist[..., -1].mean(), loc[:, -1, :])


if __name__ == '__main__':
  absltest.main()
