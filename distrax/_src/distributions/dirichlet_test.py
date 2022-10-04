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
"""Tests for `dirichlet.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import beta
from distrax._src.distributions import dirichlet
from distrax._src.utils import equivalence
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class DirichletTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(dirichlet.Dirichlet)

  @parameterized.named_parameters(
      ('1d params', (3,), ()),
      ('2d params', (4, 3), (4,)),
  )
  def test_properties(self, concentration_shape, batch_shape):
    rng = np.random.default_rng(42)
    concentration = 0.1 + rng.uniform(size=concentration_shape)
    dist = self.distrax_cls(concentration)
    self.assertEqual(dist.event_shape, (3,))
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=1e-4)(dist.concentration, concentration)

  @parameterized.named_parameters(
      ('0d params', ()),
      ('1d params with K=1', (1,)),
      ('2d params with K=1', (4, 1)),
  )
  def test_raises_on_wrong_concentration(self, concentration_shape):
    rng = np.random.default_rng(42)
    concentration = 0.1 + rng.uniform(size=concentration_shape)
    with self.assertRaises(ValueError):
      self.distrax_cls(concentration)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d params, no shape', (3,), ()),
      ('1d params, int shape', (3,), 4),
      ('1d params, 1-tuple shape', (3,), (4,)),
      ('1d params, 2-tuple shape', (3,), (5, 4)),
      ('2d params, no shape', (2, 3), ()),
      ('2d params, int shape', (2, 3), 4),
      ('2d params, 1-tuple shape', (2, 3), (4,)),
      ('2d params, 2-tuple shape', (2, 3), (5, 4)),
  )
  def test_sample_shape(self, concentration_shape, sample_shape):
    rng = np.random.default_rng(42)
    concentration = 0.1 + rng.uniform(size=concentration_shape)
    super()._test_sample_shape((concentration,), dict(), sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float16', jnp.float16))
  def test_sample_dtype(self, dtype):
    dist = self.distrax_cls(concentration=jnp.ones((3,), dtype=dtype))
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    self.assertEqual(samples.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d params, no shape', (3,), ()),
      ('1d params, int shape', (3,), 4),
      ('1d params, 1-tuple shape', (3,), (4,)),
      ('1d params, 2-tuple shape', (3,), (5, 4)),
      ('2d params, no shape', (2, 3), ()),
      ('2d params, int shape', (2, 3), 4),
      ('2d params, 1-tuple shape', (2, 3), (4,)),
      ('2d params, 2-tuple shape', (2, 3), (5, 4)),
  )
  def test_sample_and_log_prob(self, concentration_shape, sample_shape):
    rng = np.random.default_rng(42)
    concentration = 0.1 + rng.uniform(size=concentration_shape)
    super()._test_sample_and_log_prob(
        dist_args=(concentration,),
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d params, 1d value', (3,), (3,)),
      ('1d params, 2d value', (3,), (4, 3)),
      ('2d params, 1d value', (4, 3), (3,)),
      ('2d params, 2d value', (4, 3), (4, 3)),
      ('2d params, 3d value', (4, 3), (5, 4, 3)),
  )
  def test_methods_with_value(self, concentration_shape, value_shape):
    rng = np.random.default_rng(42)
    concentration = np.abs(rng.normal(size=concentration_shape))
    value = rng.uniform(size=value_shape)
    value /= np.sum(value, axis=-1, keepdims=True)
    for method in ['prob', 'log_prob']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=(concentration,),
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('1d params', (3,)),
      ('2d params', (4, 3)),
  )
  def test_method(self, concentration_shape):
    rng = np.random.default_rng(42)
    concentration = np.abs(rng.normal(size=concentration_shape))
    for method in ['entropy', 'mean', 'variance', 'stddev', 'covariance']:
      with self.subTest(method=method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=(concentration,),
            assertion_fn=self.assertion_fn(rtol=1e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('without nans', (2., 3., 4.), (1. / 6., 2. / 6., 3. / 6.)),
      ('with nans', (0.5, 3., 4.), np.nan),
  )
  def test_mode(self, concentration, expected_result):
    dist = self.distrax_cls(concentration)
    result = self.variant(dist.mode)()
    if np.any(np.isnan(expected_result)):
      self.assertTrue(jnp.all(jnp.isnan(result)))
    else:
      self.assertion_fn(rtol=1e-3)(result, expected_result)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('distrax_to_distrax', 'distrax_to_distrax'),
      ('distrax_to_tfp', 'distrax_to_tfp'),
      ('tfp_to_distrax', 'tfp_to_distrax'),
  )
  def test_with_two_distributions(self, mode_string):
    rng = np.random.default_rng(42)
    concentration1 = np.abs(rng.normal(size=(4, 3, 2))).astype(np.float32)
    concentration2 = np.abs(rng.normal(size=(3, 2))).astype(np.float32)
    for method in ['kl_divergence', 'cross_entropy']:
      with self.subTest(method=method):
        super()._test_with_two_distributions(
            attribute_string=method,
            mode_string=mode_string,
            dist1_kwargs={'concentration': concentration1},
            dist2_kwargs={'concentration': concentration2},
            assertion_fn=self.assertion_fn(rtol=3e-2))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('distrax_to_distrax', 'distrax_to_distrax'),
      ('distrax_to_tfp', 'distrax_to_tfp'),
      ('tfp_to_distrax', 'tfp_to_distrax'),
  )
  def test_with_two_distributions_dirichlet_beta(self, mode_string):
    rng = np.random.default_rng(42)
    # Distribution 1 is Dirichlet.
    alpha1 = rng.uniform(size=(4, 3)).astype(np.float32)
    beta1 = rng.uniform(size=(4, 3)).astype(np.float32)
    concentration1 = np.stack((alpha1, beta1), axis=-1)
    distrax_dist1 = self.distrax_cls(concentration1)
    tfp_dist1 = tfd.Dirichlet(concentration1)
    # Distribution 2 is Beta.
    alpha2 = rng.uniform(size=(3,)).astype(np.float32)
    beta2 = rng.uniform(size=(3,)).astype(np.float32)
    distrax_dist2 = beta.Beta(alpha2, beta2)
    tfp_dist2 = tfd.Beta(alpha2, beta2)
    for method in ['kl_divergence', 'cross_entropy']:
      with self.subTest(method=method):
        # Expected results are computed using TFP Beta-to-Beta KL divergence.
        expected_result_1 = getattr(tfd.Beta(alpha1, beta1), method)(tfp_dist2)
        expected_result_2 = getattr(tfp_dist2, method)(tfd.Beta(alpha1, beta1))
        if mode_string == 'distrax_to_distrax':
          result1 = self.variant(getattr(distrax_dist1, method))(distrax_dist2)
          result2 = self.variant(getattr(distrax_dist2, method))(distrax_dist1)
        elif mode_string == 'distrax_to_tfp':
          result1 = self.variant(getattr(distrax_dist1, method))(tfp_dist2)
          result2 = self.variant(getattr(distrax_dist2, method))(tfp_dist1)
        elif mode_string == 'tfp_to_distrax':
          result1 = self.variant(getattr(tfp_dist1, method))(distrax_dist2)
          result2 = self.variant(getattr(tfp_dist2, method))(distrax_dist1)
        self.assertion_fn(rtol=3e-2)(result1, expected_result_1)
        self.assertion_fn(rtol=3e-2)(result2, expected_result_2)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('dirichlet_dirichlet', dirichlet.Dirichlet),
      ('beta_dirichlet', beta.Beta),
  )
  def test_kl_raises_on_wrong_dims(self, dist2_type):
    rng = np.random.default_rng(42)
    # Distribution 1 is Dirichlet.
    concentration1 = np.abs(rng.normal(size=(5, 3)))
    dist1 = self.distrax_cls(concentration1)
    # Distribution 2 is either Dirichlet or Beta.
    if dist2_type is dirichlet.Dirichlet:
      dist2_kwargs = {'concentration': rng.uniform(size=(5, 4))}
    elif dist2_type is beta.Beta:
      dist2_kwargs = {'alpha': rng.uniform(size=(5,)),
                      'beta': rng.uniform(size=(5,))}
    dist2 = dist2_type(**dist2_kwargs)
    with self.assertRaises(ValueError):
      self.variant(dist1.kl_divergence)(dist2)
    with self.assertRaises(ValueError):
      self.variant(dist2.kl_divergence)(dist1)

  def test_jitable(self):
    rng = np.random.default_rng(42)
    concentration = np.abs(rng.normal(size=(4,)))
    super()._test_jittable(
        (concentration,), assertion_fn=self.assertion_fn(rtol=1e-3))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(42)
    concentration = np.abs(rng.normal(size=(6, 5, 4)))
    dist = self.distrax_cls(concentration)
    self.assertIsInstance(dist, self.distrax_cls)
    self.assertion_fn(rtol=1e-3)(
        dist[slice_].concentration, concentration[slice_])

  def test_slice_ellipsis(self):
    rng = np.random.default_rng(42)
    concentration = np.abs(rng.normal(size=(6, 5, 4)))
    dist = self.distrax_cls(concentration)
    self.assertIsInstance(dist, self.distrax_cls)
    self.assertion_fn(rtol=1e-3)(
        dist[..., -1].concentration, concentration[:, -1, :])


if __name__ == '__main__':
  absltest.main()
