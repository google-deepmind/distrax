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
"""Tests for `mvn_tri.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions.mvn_tri import MultivariateNormalTri
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np


def _get_scale_tril_from_scale_triu(scale_triu: np.ndarray) -> np.ndarray:
  scale_triu = np.triu(scale_triu)
  scale_triu_t = np.vectorize(np.transpose, signature='(k,k)->(k,k)')(
      scale_triu)
  cov = np.matmul(scale_triu, scale_triu_t)
  return np.linalg.cholesky(cov)


class MultivariateNormalTriTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(MultivariateNormalTri)

  @parameterized.named_parameters(
      ('all inputs are None', {}),
      ('wrong dimension of loc', {
          'loc': np.array(0.),
      }),
      ('scale_tri is 0d', {
          'scale_tri': np.array(1.),
      }),
      ('scale_tri is 1d', {
          'scale_tri': np.ones((4,)),
      }),
      ('scale_tri is not square', {
          'scale_tri': np.ones((4, 3)),
      }),
      ('inconsistent loc and scale_tri', {
          'loc': np.zeros((4,)),
          'scale_tri': np.ones((5, 5)),
      }),
  )
  def test_raises_on_wrong_inputs(self, dist_kwargs):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_kwargs)

  @parameterized.named_parameters(
      ('loc provided', {'loc': np.zeros((4,))}),
      ('scale_tri provided', {'scale_tri': np.eye(4)}),
  )
  def test_default_properties(self, dist_kwargs):
    dist = self.distrax_cls(**dist_kwargs)
    self.assertTrue(dist.is_lower)
    self.assertion_fn(rtol=1e-3)(dist.loc, jnp.zeros((4,)))
    self.assertion_fn(rtol=1e-3)(dist.scale_tri, jnp.eye(4))

  @parameterized.named_parameters(
      ('unbatched', (), (4,), (4, 4), True),
      ('batched loc', (7,), (7, 4), (4, 4), True),
      ('batched scale_tri lower', (7,), (4,), (7, 4, 4), True),
      ('batched scale_tri upper', (7,), (4,), (7, 4, 4), False),
  )
  def test_properties(self, batch_shape, loc_shape, scale_tri_shape, is_lower):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_tri = rng.normal(size=scale_tri_shape)
    dist = self.distrax_cls(loc=loc, scale_tri=scale_tri, is_lower=is_lower)
    tri_fn = jnp.tril if is_lower else jnp.triu
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertEqual(dist.is_lower, is_lower)
    self.assertion_fn(rtol=1e-3)(
        dist.loc, jnp.broadcast_to(loc, batch_shape + (4,)))
    self.assertion_fn(rtol=1e-3)(dist.scale_tri, jnp.broadcast_to(
        tri_fn(scale_tri), batch_shape + (4, 4)))

  @chex.all_variants
  @parameterized.named_parameters(
      ('unbatched, no shape', (), (4,), (4, 4)),
      ('batched loc, no shape', (), (7, 4), (4, 4)),
      ('batched scale_tri, no shape', (), (4,), (7, 4, 4)),
      ('unbatched, with shape', (3,), (4,), (4, 4)),
      ('batched loc, with shape', (3,), (7, 4), (4, 4)),
      ('batched scale_tri, with shape', (3,), (4,), (7, 4, 4)),
  )
  def test_sample_shape(self, sample_shape, loc_shape, scale_tri_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_tri = rng.normal(size=scale_tri_shape)
    dist_kwargs = {'loc': loc, 'scale_tri': scale_tri}
    tfp_dist_kwargs = {'loc': loc, 'scale_tril': scale_tri}
    super()._test_sample_shape(
        dist_args=(), dist_kwargs=dist_kwargs, tfp_dist_kwargs=tfp_dist_kwargs,
        sample_shape=sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist_params = {
        'loc': np.array([0., 0.], dtype),
        'scale_tri': np.array([[1., 0.], [0., 1.]], dtype)}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('unbatched, unbatched value', (4,), (4,), (4, 4), True),
      ('unbatched, unbatched value, upper', (4,), (4,), (4, 4), False),
      ('batched loc, unbatched value', (4,), (7, 4), (4, 4), True),
      ('batched scale_tri, unbatched value', (4,), (4,), (7, 4, 4), True),
      ('unbatched, batched value', (3, 7, 4), (4,), (4, 4), True),
      ('batched loc, batched value', (3, 7, 4), (7, 4), (4, 4), True),
      ('batched scale_tri, batched value', (3, 7, 4), (4,), (7, 4, 4), True),
      ('batched scale_tri, batched value, upper',
       (3, 7, 4), (4,), (7, 4, 4), False),
  )
  def test_log_prob(self, value_shape, loc_shape, scale_tri_shape, is_lower):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_tri = rng.normal(size=scale_tri_shape)
    value = rng.normal(size=value_shape)
    dist_kwargs = {'loc': loc, 'scale_tri': scale_tri, 'is_lower': is_lower}
    if is_lower:
      tfp_dist_kwargs = {'loc': loc, 'scale_tril': scale_tri}
    else:
      scale_tril = _get_scale_tril_from_scale_triu(scale_tri)
      tfp_dist_kwargs = {'loc': loc, 'scale_tril': scale_tril}
    super()._test_attribute(
        attribute_string='log_prob',
        dist_kwargs=dist_kwargs,
        tfp_dist_kwargs=tfp_dist_kwargs,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('unbatched', (4,), (4, 4)),
      ('batched loc', (7, 4), (4, 4)),
      ('batched scale_tri', (4,), (7, 4, 4)),
  )
  def test_method(self, loc_shape, scale_tri_shape):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=loc_shape)
    scale_tri = rng.normal(size=scale_tri_shape)
    for method in ['entropy', 'mean', 'stddev', 'variance',
                   'covariance', 'mode']:
      for is_lower in [True, False]:
        if method in ['stddev', 'covariance', 'variance']:
          rtol = 2e-2 if is_lower else 5e-2
        else:
          rtol = 1e-3

        dist_kwargs = {'loc': loc, 'scale_tri': scale_tri, 'is_lower': is_lower}
        if is_lower:
          tfp_dist_kwargs = {'loc': loc, 'scale_tril': scale_tri}
        else:
          scale_tril = _get_scale_tril_from_scale_triu(scale_tri)
          tfp_dist_kwargs = {'loc': loc, 'scale_tril': scale_tril}
        with self.subTest(method=method, is_lower=is_lower):
          super()._test_attribute(
              method,
              dist_kwargs=dist_kwargs,
              tfp_dist_kwargs=tfp_dist_kwargs,
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
    loc1 = rng.normal(size=(5, 1, 4))
    scale_tri1 = rng.normal(size=(3, 4, 4))
    loc2 = rng.normal(size=(3, 4))
    scale_tri2 = rng.normal(size=(4, 4))
    for is_lower in [True, False]:
      dist1_kwargs = {
          'loc': loc1, 'scale_tri': scale_tri1, 'is_lower': is_lower}
      dist2_kwargs = {
          'loc': loc2, 'scale_tri': scale_tri2, 'is_lower': is_lower}
      if is_lower:
        tfp_dist1_kwargs = {'loc': loc1, 'scale_tril': scale_tri1}
        tfp_dist2_kwargs = {'loc': loc2, 'scale_tril': scale_tri2}
      else:
        tfp_dist1_kwargs = {
            'loc': loc1,
            'scale_tril': _get_scale_tril_from_scale_triu(scale_tri1)
        }
        tfp_dist2_kwargs = {
            'loc': loc2,
            'scale_tril': _get_scale_tril_from_scale_triu(scale_tri2)
        }
      with self.subTest(is_lower=is_lower):
        super()._test_with_two_distributions(
            attribute_string=function_string,
            mode_string=mode_string,
            dist1_kwargs=dist1_kwargs,
            dist2_kwargs=dist2_kwargs,
            tfp_dist1_kwargs=tfp_dist1_kwargs,
            tfp_dist2_kwargs=tfp_dist2_kwargs,
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
    scale_tri = rng.normal(size=(4, 4))
    for is_lower in [True, False]:
      with self.subTest(is_lower=is_lower):
        dist_kwargs = {'loc': loc, 'scale_tri': scale_tri, 'is_lower': is_lower}
        dist = self.distrax_cls(**dist_kwargs)
        self.assertEqual(dist[slice_].batch_shape, loc[slice_].shape[:-1])
        self.assertEqual(dist[slice_].event_shape, dist.event_shape)
        self.assertEqual(dist[slice_].is_lower, dist.is_lower)
        self.assertion_fn(rtol=1e-3)(dist[slice_].mean(), loc[slice_])

  def test_slice_ellipsis(self):
    rng = np.random.default_rng(2022)
    loc = rng.normal(size=(6, 5, 4))
    scale_tri = rng.normal(size=(4, 4))
    for is_lower in [True, False]:
      with self.subTest(is_lower=is_lower):
        dist_kwargs = {'loc': loc, 'scale_tri': scale_tri, 'is_lower': is_lower}
        dist = self.distrax_cls(**dist_kwargs)
        self.assertEqual(dist[..., -1].batch_shape, (6,))
        self.assertEqual(dist[..., -1].event_shape, dist.event_shape)
        self.assertEqual(dist[..., -1].is_lower, dist.is_lower)
        self.assertion_fn(rtol=1e-3)(dist[..., -1].mean(), loc[:, -1, :])


if __name__ == '__main__':
  absltest.main()
