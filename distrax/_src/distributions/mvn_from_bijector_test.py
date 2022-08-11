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
"""Tests for `mvn_from_bijector.py`."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import linear
from distrax._src.bijectors.diag_linear import DiagLinear
from distrax._src.bijectors.triangular_linear import TriangularLinear
from distrax._src.distributions.mvn_from_bijector import MultivariateNormalFromBijector

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array


class MockLinear(linear.Linear):
  """A mock linear bijector."""

  def __init__(self, event_dims: int):
    super().__init__(event_dims, batch_shape=(), dtype=float)

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return x, jnp.zeros_like(x)[:-1]


class MultivariateNormalFromBijectorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('loc is 0d', 4, np.zeros(shape=())),
      ('loc and scale dims not compatible', 3, np.zeros((4,))),
  )
  def test_raises_on_wrong_inputs(self, event_dims, loc):
    bij = MockLinear(event_dims)
    with self.assertRaises(ValueError):
      MultivariateNormalFromBijector(loc, bij)

  @parameterized.named_parameters(
      ('no broadcast', np.ones((4,)), np.zeros((4,)), (4,)),
      ('broadcasted loc', np.ones((3, 4)), np.zeros((4,)), (3, 4)),
      ('broadcasted diag', np.ones((4,)), np.zeros((3, 4)), (3, 4)),
  )
  def test_loc_scale_and_shapes(self, diag, loc, expected_shape):
    scale = DiagLinear(diag)
    batch_shape = jnp.broadcast_shapes(diag.shape, loc.shape)[:-1]
    dist = MultivariateNormalFromBijector(loc, scale)
    np.testing.assert_allclose(dist.loc, np.zeros(expected_shape))
    self.assertTrue(scale.same_as(dist.scale))
    self.assertEqual(dist.event_shape, (4,))
    self.assertEqual(dist.batch_shape, batch_shape)

  @chex.all_variants
  def test_sample(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = 0.5 + jax.random.uniform(next(prng), (4,))
    loc = jax.random.normal(next(prng), (4,))
    scale = DiagLinear(diag)
    dist = MultivariateNormalFromBijector(loc, scale)
    num_samples = 100_000
    sample_fn = lambda seed: dist.sample(seed=seed, sample_shape=num_samples)
    samples = self.variant(sample_fn)(jax.random.PRNGKey(2000))
    self.assertEqual(samples.shape, (num_samples, 4))
    np.testing.assert_allclose(jnp.mean(samples, axis=0), loc, rtol=0.1)
    np.testing.assert_allclose(jnp.std(samples, axis=0), diag, rtol=0.1)

  @chex.all_variants
  def test_log_prob(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = 0.5 + jax.random.uniform(next(prng), (4,))
    loc = jax.random.normal(next(prng), (4,))
    scale = DiagLinear(diag)
    dist = MultivariateNormalFromBijector(loc, scale)
    values = jax.random.normal(next(prng), (5, 4))
    tfp_dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=diag)
    np.testing.assert_allclose(
        self.variant(dist.log_prob)(values), tfp_dist.log_prob(values),
        rtol=2e-7)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('no broadcast', (4,), (4,)),
      ('broadcasted loc', (3, 4), (4,)),
      ('broadcasted diag', (4,), (3, 4)),
  )
  def test_mean_median_mode(self, diag_shape, loc_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    diag = jax.random.normal(next(prng), diag_shape)
    loc = jax.random.normal(next(prng), loc_shape)
    scale = DiagLinear(diag)
    batch_shape = jnp.broadcast_shapes(diag_shape, loc_shape)[:-1]
    dist = MultivariateNormalFromBijector(loc, scale)
    for method in ['mean', 'median', 'mode']:
      with self.subTest(method=method):
        fn = self.variant(getattr(dist, method))
        np.testing.assert_allclose(
            fn(), jnp.broadcast_to(loc, batch_shape + loc.shape[-1:]))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('no broadcast', (4,), (4,)),
      ('broadcasted loc', (3, 4), (4,)),
      ('broadcasted diag', (4,), (3, 4)),
  )
  def test_variance_stddev_covariance_diag(self, scale_shape, loc_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    scale_diag = jax.random.normal(next(prng), scale_shape)
    loc = jax.random.normal(next(prng), loc_shape)
    scale = DiagLinear(scale_diag)
    batch_shape = jnp.broadcast_shapes(scale_shape[:-1], loc_shape[:-1])
    dist = MultivariateNormalFromBijector(loc, scale)
    for method in ['variance', 'stddev', 'covariance']:
      with self.subTest(method=method):
        fn = self.variant(getattr(dist, method))
        if method == 'variance':
          expected_result = jnp.broadcast_to(
              jnp.square(scale_diag), batch_shape + loc.shape[-1:])
        elif method == 'stddev':
          expected_result = jnp.broadcast_to(
              jnp.abs(scale_diag), batch_shape + loc.shape[-1:])
        elif method == 'covariance':
          expected_result = jnp.broadcast_to(
              jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(
                  jnp.square(scale_diag)),
              batch_shape + loc.shape[-1:] + loc.shape[-1:])
        np.testing.assert_allclose(fn(), expected_result, rtol=5e-3)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('no broadcast', (4, 4), (4,)),
      ('broadcasted loc', (3, 4, 4), (4,)),
      ('broadcasted diag', (4, 4), (3, 4)),
  )
  def test_variance_stddev_covariance_no_diag(self, scale_shape, loc_shape):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))
    scale_tril = jnp.tril(jax.random.normal(next(prng), scale_shape))
    loc = jax.random.normal(next(prng), loc_shape)
    scale = TriangularLinear(matrix=scale_tril, is_lower=True)
    batch_shape = jnp.broadcast_shapes(scale_shape[:-2], loc_shape[:-1])
    dist = MultivariateNormalFromBijector(loc, scale)
    for method in ['variance', 'stddev', 'covariance']:
      with self.subTest(method=method):
        fn = self.variant(getattr(dist, method))
        scale_tril_t = jnp.vectorize(
            jnp.transpose, signature='(k,k)->(k,k)')(scale_tril)
        scale_times_scale_t = jnp.matmul(scale_tril, scale_tril_t)
        if method == 'variance':
          expected_result = jnp.vectorize(jnp.diag, signature='(k,k)->(k)')(
              scale_times_scale_t)
          expected_result = jnp.broadcast_to(
              expected_result, batch_shape + loc.shape[-1:])
        elif method == 'stddev':
          expected_result = jnp.vectorize(jnp.diag, signature='(k,k)->(k)')(
              jnp.sqrt(scale_times_scale_t))
          expected_result = jnp.broadcast_to(
              expected_result, batch_shape + loc.shape[-1:])
        elif method == 'covariance':
          expected_result = jnp.broadcast_to(
              scale_times_scale_t, batch_shape + scale_tril.shape[-2:])
        np.testing.assert_allclose(fn(), expected_result, rtol=5e-3)

  @chex.all_variants(with_pmap=False)
  def test_kl_divergence_diag_distributions(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))

    scale_diag1 = 0.1 + jax.random.uniform(next(prng), (3, 4))
    loc1 = jax.random.normal(next(prng), (1, 4))
    dist1_distrax = MultivariateNormalFromBijector(
        loc=loc1,
        scale=DiagLinear(scale_diag1),
    )
    dist1_tfp = tfd.MultivariateNormalDiag(
        loc=loc1, scale_diag=scale_diag1)

    scale_diag2 = 0.1 + jax.random.uniform(next(prng), (4,))
    loc2 = jax.random.normal(next(prng), (4,))
    dist2_distrax = MultivariateNormalFromBijector(
        loc=loc2,
        scale=DiagLinear(scale_diag2),
    )
    dist2_tfp = tfd.MultivariateNormalDiag(
        loc=loc2, scale_diag=scale_diag2)

    expected_result1 = dist1_tfp.kl_divergence(dist2_tfp)
    expected_result2 = dist2_tfp.kl_divergence(dist1_tfp)

    for mode in ['distrax_to_distrax', 'distrax_to_tfp', 'tfp_to_distrax']:
      with self.subTest(mode=mode):
        if mode == 'distrax_to_distrax':
          result1 = self.variant(dist1_distrax.kl_divergence)(dist2_distrax)
          result2 = self.variant(dist2_distrax.kl_divergence)(dist1_distrax)
        elif mode == 'distrax_to_tfp':
          result1 = self.variant(dist1_distrax.kl_divergence)(dist2_tfp)
          result2 = self.variant(dist2_distrax.kl_divergence)(dist1_tfp)
        elif mode == 'tfp_to_distrax':
          result1 = self.variant(dist1_tfp.kl_divergence)(dist2_distrax)
          result2 = self.variant(dist2_tfp.kl_divergence)(dist1_distrax)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-3)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-3)

  @chex.all_variants(with_pmap=False)
  def test_kl_divergence_non_diag_distributions(self):
    prng = hk.PRNGSequence(jax.random.PRNGKey(42))

    scale_tril1 = jnp.tril(jax.random.normal(next(prng), (3, 4, 4)))
    loc1 = jax.random.normal(next(prng), (1, 4))
    dist1_distrax = MultivariateNormalFromBijector(
        loc=loc1,
        scale=TriangularLinear(matrix=scale_tril1),
    )
    dist1_tfp = tfd.MultivariateNormalTriL(loc=loc1, scale_tril=scale_tril1)

    scale_tril2 = jnp.tril(jax.random.normal(next(prng), (4, 4)))
    loc2 = jax.random.normal(next(prng), (4,))
    dist2_distrax = MultivariateNormalFromBijector(
        loc=loc2,
        scale=TriangularLinear(matrix=scale_tril2),
    )
    dist2_tfp = tfd.MultivariateNormalTriL(loc=loc2, scale_tril=scale_tril2)

    expected_result1 = dist1_tfp.kl_divergence(dist2_tfp)
    expected_result2 = dist2_tfp.kl_divergence(dist1_tfp)

    for mode in ['distrax_to_distrax', 'distrax_to_tfp', 'tfp_to_distrax']:
      with self.subTest(mode=mode):
        if mode == 'distrax_to_distrax':
          result1 = self.variant(dist1_distrax.kl_divergence)(dist2_distrax)
          result2 = self.variant(dist2_distrax.kl_divergence)(dist1_distrax)
        elif mode == 'distrax_to_tfp':
          result1 = self.variant(dist1_distrax.kl_divergence)(dist2_tfp)
          result2 = self.variant(dist2_distrax.kl_divergence)(dist1_tfp)
        elif mode == 'tfp_to_distrax':
          result1 = self.variant(dist1_tfp.kl_divergence)(dist2_distrax)
          result2 = self.variant(dist2_tfp.kl_divergence)(dist1_distrax)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-3)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-3)

  def test_kl_divergence_raises_on_incompatible_distributions(self):
    dim = 4
    dist1 = MultivariateNormalFromBijector(
        loc=jnp.zeros((dim,)),
        scale=DiagLinear(diag=jnp.ones((dim,))),
    )
    dim = 5
    dist2 = MultivariateNormalFromBijector(
        loc=jnp.zeros((dim,)),
        scale=DiagLinear(diag=jnp.ones((dim,))),
    )
    with self.assertRaises(ValueError):
      dist1.kl_divergence(dist2)


if __name__ == '__main__':
  absltest.main()
