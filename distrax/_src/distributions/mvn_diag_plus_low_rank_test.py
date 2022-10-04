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
"""Tests for `mvn_diag_plus_low_rank.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from distrax._src.utils import equivalence

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _covariance_matrix_from_low_rank(
    scale_diag, scale_u_matrix, scale_v_matrix):
  """Constructs the covariance matrix from the low-rank matrices."""
  if scale_u_matrix is not None:
    if scale_v_matrix is None:
      scale_v_matrix = np.copy(scale_u_matrix)
    scale_v_matrix_t = np.vectorize(
        np.transpose, signature='(k,m)->(m,k)')(scale_v_matrix)
    scale = np.matmul(scale_u_matrix, scale_v_matrix_t) + np.vectorize(
        np.diag, signature='(k)->(k,k)')(scale_diag)
  else:
    scale = np.vectorize(np.diag, signature='(k)->(k,k)')(scale_diag)
  scale_t = np.vectorize(np.transpose, signature='(k,k)->(k,k)')(scale)
  return np.matmul(scale, scale_t)


class MultivariateNormalDiagPlusLowRankTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(MultivariateNormalDiagPlusLowRank)

  @parameterized.named_parameters(
      ('all inputs are None', {}),
      ('scale_v_matrix is provided but scale_u_matrix is None', {
          'loc': np.zeros((4,)),
          'scale_v_matrix': np.ones((4, 2)),
      }),
      ('wrong dimension of loc', {
          'loc': np.array(0.),
      }),
      ('wrong dimension of scale_diag', {
          'scale_diag': np.array(0.),
      }),
      ('wrong dimension of scale_u_matrix', {
          'scale_u_matrix': np.ones((4,)),
      }),
      ('wrong dimension of scale_v_matrix', {
          'scale_u_matrix': np.ones((4, 2)),
          'scale_v_matrix': np.ones((4,)),
      }),
      ('last dimension of scale_u_matrix is zero', {
          'scale_u_matrix': np.ones((4, 0)),
      }),
      ('inconsistent dimensions of scale_u_matrix and scale_v_matrix', {
          'scale_u_matrix': np.ones((4, 2)),
          'scale_v_matrix': np.ones((4, 1)),
      }),
      ('inconsistent event_dim across two params', {
          'loc': np.zeros((4,)),
          'scale_u_matrix': np.ones((5, 2)),
      }),
      ('inconsistent event_dim across three params', {
          'loc': np.zeros((4,)),
          'scale_diag': np.ones((5,)),
          'scale_u_matrix': np.ones((4, 2)),
      }),
  )
  def test_raises_on_wrong_inputs(self, dist_kwargs):
    with self.assertRaises(ValueError):
      MultivariateNormalDiagPlusLowRank(**dist_kwargs)

  @parameterized.named_parameters(
      ('loc provided', {'loc': np.zeros((4,))}),
      ('scale_diag provided', {'scale_diag': np.ones((4,))}),
      ('scale_u_matrix provided', {'scale_u_matrix': np.zeros((4, 1))}),
  )
  def test_default_properties(self, dist_kwargs):
    dist = MultivariateNormalDiagPlusLowRank(**dist_kwargs)
    self.assertion_fn(rtol=1e-3)(dist.loc, jnp.zeros((4,)))
    self.assertion_fn(rtol=1e-3)(dist.scale_diag, jnp.ones((4,)))
    self.assertion_fn(rtol=1e-3)(dist.scale_u_matrix, jnp.zeros((4, 1)))
    self.assertion_fn(rtol=1e-3)(dist.scale_v_matrix, jnp.zeros((4, 1)))

  @parameterized.named_parameters(
      ('unbatched', (), (4,), (4,), (4, 2), (4, 2)),
      ('batched loc', (7,), (7, 4), (4,), (4, 2), (4, 2)),
      ('batched scale_diag', (7,), (4,), (7, 4), (4, 2), (4, 2)),
      ('batched scale_u_matrix', (7,), (4,), (4,), (7, 4, 2), (4, 2)),
      ('batched scale_v_matrix', (7,), (4,), (4,), (4, 2), (7, 4, 2)),
  )
  def test_properties(self, batch_shape, loc_shape, scale_diag_shape,
                      scale_u_matrix_shape, scale_v_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_diag = rng.normal(size=scale_diag_shape)
    scale_u_matrix = rng.normal(size=scale_u_matrix_shape)
    scale_v_matrix = rng.normal(size=scale_v_matrix_shape)
    dist = MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_diag=scale_diag,
        scale_u_matrix=scale_u_matrix,
        scale_v_matrix=scale_v_matrix,
    )
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=1e-3)(
        dist.loc, jnp.broadcast_to(loc, batch_shape + (4,)))
    self.assertion_fn(rtol=1e-3)(
        dist.scale_diag, jnp.broadcast_to(scale_diag, batch_shape + (4,)))
    self.assertion_fn(rtol=1e-3)(
        dist.scale_u_matrix,
        jnp.broadcast_to(scale_u_matrix, batch_shape + (4, 2)))
    self.assertion_fn(rtol=1e-3)(
        dist.scale_v_matrix,
        jnp.broadcast_to(scale_v_matrix, batch_shape + (4, 2)))

  @chex.all_variants
  @parameterized.named_parameters(
      ('unbatched, no shape', (), (4,), (4,), (4, 2), (4, 2)),
      ('batched loc, no shape', (), (7, 4), (4,), (4, 2), (4, 2)),
      ('batched scale_diag, no shape', (), (4,), (7, 4), (4, 2), (4, 2)),
      ('batched scale_u_matrix, no shape', (), (4,), (4,), (7, 4, 2), (4, 2)),
      ('batched scale_v_matrix, no shape', (), (4,), (4,), (4, 2), (7, 4, 2)),
      ('unbatched, with shape', (3,), (4,), (4,), (4, 2), (4, 2)),
      ('batched loc, with shape', (3,), (7, 4), (4,), (4, 2), (4, 2)),
      ('batched scale_diag, with shape', (3,), (4,), (7, 4), (4, 2), (4, 2)),
      ('batched scale_u_matrix, with shape',
       (3,), (4,), (4,), (7, 4, 2), (4, 2)),
      ('batched scale_v_matrix, with shape',
       (3,), (4,), (4,), (4, 2), (7, 4, 2)),
  )
  def test_sample_shape(self, sample_shape, loc_shape, scale_diag_shape,
                        scale_u_matrix_shape, scale_v_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_diag = rng.normal(size=scale_diag_shape)
    scale_u_matrix = rng.normal(size=scale_u_matrix_shape)
    scale_v_matrix = rng.normal(size=scale_v_matrix_shape)
    dist = MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_diag=scale_diag,
        scale_u_matrix=scale_u_matrix,
        scale_v_matrix=scale_v_matrix,
    )
    tfp_dist = tfd.MultivariateNormalFullCovariance(
        loc=loc,
        covariance_matrix=_covariance_matrix_from_low_rank(
            scale_diag, scale_u_matrix, scale_v_matrix)
    )
    sample_fn = self.variant(
        lambda rng: dist.sample(sample_shape=sample_shape, seed=rng))
    distrax_samples = sample_fn(jax.random.PRNGKey(0))
    tfp_samples = tfp_dist.sample(
        sample_shape=sample_shape, seed=jax.random.PRNGKey(0))
    self.assertEqual(distrax_samples.shape, tfp_samples.shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist_params = {
        'loc': np.array([0., 0.], dtype),
        'scale_diag': np.array([1., 1.], dtype)}
    dist = MultivariateNormalDiagPlusLowRank(**dist_params)
    samples = self.variant(dist.sample)(seed=jax.random.PRNGKey(0))
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('unbatched, unbatched value', (4,), (4,), (4,), (4, 2), (4, 2)),
      ('batched loc, unbatched value', (4,), (7, 4), (4,), (4, 2), (4, 2)),
      ('batched scale_diag, unbatched value',
       (4,), (4,), (7, 4), (4, 2), (4, 2)),
      ('batched scale_u_matrix, unbatched value',
       (4,), (4,), (4,), (7, 4, 2), (4, 2)),
      ('batched scale_v_matrix, unbatched value',
       (4,), (4,), (4,), (4, 2), (7, 4, 2)),
      ('unbatched, batched value', (7, 4), (4,), (4,), (4, 2), (4, 2)),
      ('batched loc, batched value', (7, 4), (7, 4), (4,), (4, 2), (4, 2)),
      ('batched scale_diag, batched value',
       (7, 4), (4,), (7, 4), (4, 2), (4, 2)),
      ('batched scale_u_matrix, batched value',
       (7, 4), (4,), (4,), (7, 4, 2), (4, 2)),
      ('batched scale_v_matrix, batched value',
       (7, 4), (4,), (4,), (4, 2), (7, 4, 2)),
  )
  def test_log_prob(self, value_shape, loc_shape, scale_diag_shape,
                    scale_u_matrix_shape, scale_v_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_diag = rng.normal(size=scale_diag_shape)
    scale_u_matrix = 0.1 * rng.normal(size=scale_u_matrix_shape)
    scale_v_matrix = 0.1 * rng.normal(size=scale_v_matrix_shape)
    dist = MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_diag=scale_diag,
        scale_u_matrix=scale_u_matrix,
        scale_v_matrix=scale_v_matrix,
    )
    tfp_dist = tfd.MultivariateNormalFullCovariance(
        loc=loc,
        covariance_matrix=_covariance_matrix_from_low_rank(
            scale_diag, scale_u_matrix, scale_v_matrix)
    )
    value = rng.normal(size=value_shape)
    self.assertion_fn(rtol=2e-3)(
        self.variant(dist.log_prob)(value), tfp_dist.log_prob(value))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('unbatched', (4,), (4,), (4, 2), (4, 2)),
      ('batched loc', (7, 4), (4,), (4, 2), (4, 2)),
      ('batched scale_diag', (4,), (7, 4), (4, 2), (4, 2)),
      ('batched scale_u_matrix', (4,), (4,), (7, 4, 2), (4, 2)),
      ('batched scale_v_matrix', (4,), (4,), (4, 2), (7, 4, 2)),
      ('scale_u_matrix is None', (4,), (4,), None, None),
      ('scale_v_matrix is None', (4,), (4,), (4, 2), None),
  )
  def test_method(self, loc_shape, scale_diag_shape,
                  scale_u_matrix_shape, scale_v_matrix_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_diag = rng.normal(size=scale_diag_shape)
    if scale_u_matrix_shape is None:
      scale_u_matrix = None
    else:
      scale_u_matrix = 0.1 * rng.normal(size=scale_u_matrix_shape)
    if scale_v_matrix_shape is None:
      scale_v_matrix = None
    else:
      scale_v_matrix = 0.1 * rng.normal(size=scale_v_matrix_shape)
    dist = MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_diag=scale_diag,
        scale_u_matrix=scale_u_matrix,
        scale_v_matrix=scale_v_matrix,
    )
    tfp_dist = tfd.MultivariateNormalFullCovariance(
        loc=loc,
        covariance_matrix=_covariance_matrix_from_low_rank(
            scale_diag, scale_u_matrix, scale_v_matrix)
    )
    for method in [
        'entropy', 'mean', 'stddev', 'variance', 'covariance', 'mode']:
      if method in ['stddev', 'variance']:
        rtol = 1e-2
      elif method in ['covariance']:
        rtol = 8e-2
      else:
        rtol = 1e-3
      with self.subTest(method=method):
        fn = self.variant(getattr(dist, method))
        self.assertion_fn(rtol=rtol)(fn(), getattr(tfp_dist, method)())

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
    loc1 = rng.normal(size=(5, 1, 4))
    scale_diag1 = rng.normal(size=(3, 4))
    scale_u_matrix1 = 0.1 * rng.normal(size=(5, 1, 4, 2))
    scale_perturb_diag1 = rng.normal(size=(5, 1, 2))
    scale_v_matrix1 = scale_u_matrix1 * np.expand_dims(
        scale_perturb_diag1, axis=-2)
    loc2 = rng.normal(size=(3, 4))
    scale_diag2 = rng.normal(size=(3, 4))
    scale_u_matrix2 = 0.1 * rng.normal(size=(4, 2))
    scale_perturb_diag2 = rng.normal(size=(2,))
    scale_v_matrix2 = scale_u_matrix2 * np.expand_dims(
        scale_perturb_diag2, axis=-2)
    distrax_dist1 = MultivariateNormalDiagPlusLowRank(
        loc=loc1,
        scale_diag=scale_diag1,
        scale_u_matrix=scale_u_matrix1,
        scale_v_matrix=scale_v_matrix1,
    )
    distrax_dist2 = MultivariateNormalDiagPlusLowRank(
        loc=loc2,
        scale_diag=scale_diag2,
        scale_u_matrix=scale_u_matrix2,
        scale_v_matrix=scale_v_matrix2,
    )
    tfp_dist1 = tfd.MultivariateNormalDiagPlusLowRank(
        loc=loc1,
        scale_diag=scale_diag1,
        scale_perturb_factor=scale_u_matrix1,
        scale_perturb_diag=scale_perturb_diag1,
    )
    tfp_dist2 = tfd.MultivariateNormalDiagPlusLowRank(
        loc=loc2,
        scale_diag=scale_diag2,
        scale_perturb_factor=scale_u_matrix2,
        scale_perturb_diag=scale_perturb_diag2,
    )
    expected_result1 = getattr(tfp_dist1, function_string)(tfp_dist2)
    expected_result2 = getattr(tfp_dist2, function_string)(tfp_dist1)
    if mode_string == 'distrax_to_distrax':
      result1 = self.variant(getattr(distrax_dist1, function_string))(
          distrax_dist2)
      result2 = self.variant(getattr(distrax_dist2, function_string))(
          distrax_dist1)
    elif mode_string == 'distrax_to_tfp':
      result1 = self.variant(getattr(distrax_dist1, function_string))(tfp_dist2)
      result2 = self.variant(getattr(distrax_dist2, function_string))(tfp_dist1)
    elif mode_string == 'tfp_to_distrax':
      result1 = self.variant(getattr(tfp_dist1, function_string))(distrax_dist2)
      result2 = self.variant(getattr(tfp_dist2, function_string))(distrax_dist1)
    self.assertion_fn(rtol=3e-3)(result1, expected_result1)
    self.assertion_fn(rtol=3e-3)(result2, expected_result2)

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
    scale_diag = rng.normal(size=(4,))
    scale_u_matrix = rng.normal(size=(1, 4, 2))
    scale_v_matrix = rng.normal(size=(4, 2))
    dist = MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_diag=scale_diag,
        scale_u_matrix=scale_u_matrix,
        scale_v_matrix=scale_v_matrix,
    )
    self.assertEqual(dist[slice_].batch_shape, loc[slice_].shape[:-1])
    self.assertEqual(dist[slice_].event_shape, loc[slice_].shape[-1:])
    self.assertion_fn(rtol=1e-3)(dist[slice_].mean(), loc[slice_])

  def test_slice_ellipsis(self):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=(6, 5, 4))
    scale_diag = rng.normal(size=(4,))
    scale_u_matrix = rng.normal(size=(1, 4, 2))
    scale_v_matrix = rng.normal(size=(4, 2))
    dist = MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_diag=scale_diag,
        scale_u_matrix=scale_u_matrix,
        scale_v_matrix=scale_v_matrix,
    )
    self.assertEqual(dist[..., -1].batch_shape, (6,))
    self.assertEqual(dist[..., -1].event_shape, dist.event_shape)
    self.assertion_fn(rtol=1e-3)(dist[..., -1].mean(), loc[:, -1, :])


if __name__ == '__main__':
  absltest.main()
