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
"""Tests for `distribution_from_tfp.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.distribution_from_tfp import distribution_from_tfp
from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.normal import Normal
from distrax._src.distributions.transformed import Transformed
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class DistributionFromTfpNormal(parameterized.TestCase):
  """Tests for normal distribution."""

  def setUp(self):
    super().setUp()
    self._sample_shape = (np.int32(10),)
    self._seed = 42
    self._key = jax.random.PRNGKey(self._seed)
    self.base_dist = tfd.Normal(loc=0., scale=1.)
    self.values = jnp.array([1., -1.])
    self.distrax_second_dist = Normal(loc=-1., scale=0.8)
    self.tfp_second_dist = tfd.Normal(loc=-1., scale=0.8)

  def assertion_fn(self, rtol):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  @property
  def wrapped_dist(self):
    return distribution_from_tfp(self.base_dist)

  def test_event_shape(self):
    chex.assert_equal(self.wrapped_dist.event_shape, self.base_dist.event_shape)

  def test_batch_shape(self):
    chex.assert_equal(self.wrapped_dist.batch_shape, self.base_dist.batch_shape)

  @chex.all_variants
  def test_sample_dtype(self):
    samples = self.variant(self.wrapped_dist.sample)(seed=self._key)
    self.assertEqual(self.wrapped_dist.dtype, samples.dtype)
    self.assertEqual(self.wrapped_dist.dtype, self.base_dist.dtype)

  @chex.all_variants
  def test_sample(self):
    def sample_fn(key):
      return self.wrapped_dist.sample(sample_shape=self._sample_shape, seed=key)
    self.assertion_fn(rtol=2e-4)(
        self.variant(sample_fn)(self._key),
        self.base_dist.sample(sample_shape=self._sample_shape, seed=self._key))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('mean', 'mean'),
      ('mode', 'mode'),
      ('median', 'median'),
      ('stddev', 'stddev'),
      ('variance', 'variance'),
      ('entropy', 'entropy'),
  )
  def test_method(self, method):
    self.variant(lambda: None)  # To avoid variants usage error.
    try:
      expected_result = getattr(self.base_dist, method)()
    except NotImplementedError:
      return
    except AttributeError:
      return
    result = getattr(self.wrapped_dist, method)()
    self.assertion_fn(rtol=2e-4)(result, expected_result)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('log_prob', 'log_prob'),
      ('prob', 'prob'),
      ('log_cdf', 'log_cdf'),
      ('cdf', 'cdf'),
  )
  def test_method_with_value(self, method):
    self.variant(lambda: None)  # To avoid variants usage error.

    if (isinstance(self.base_dist, tfd.Categorical) and
        method in ('cdf', 'log_cdf')):
      # TODO(budden): make .cdf() and .log_cdf() from tfp.Categorical jittable.
      return

    try:
      expected_result = getattr(self.base_dist, method)(self.values)
    except NotImplementedError:
      return
    except AttributeError:
      return
    result = self.variant(getattr(self.wrapped_dist, method))(self.values)
    self.assertion_fn(rtol=2e-4)(result, expected_result)

  @chex.all_variants
  def test_sample_and_log_prob(self):
    base_samples = self.base_dist.sample(
        sample_shape=self._sample_shape, seed=self._key)
    base_logprob = self.base_dist.log_prob(base_samples)

    def sample_fn(key):
      return self.wrapped_dist.sample_and_log_prob(
          sample_shape=self._sample_shape, seed=key)
    samples, log_prob = self.variant(sample_fn)(self._key)
    self.assertion_fn(rtol=2e-4)(samples, base_samples)
    self.assertion_fn(rtol=2e-4)(log_prob, base_logprob)

  @chex.all_variants
  @parameterized.named_parameters(
      ('kl_divergence', 'kl_divergence'),
      ('cross_entropy', 'cross_entropy'),
  )
  def test_with_two_distributions(self, method):
    """Test methods of the forms listed below.

      D(distrax_distrib || wrapped_distrib),
      D(wrapped_distrib || distrax_distrib),
      D(tfp_distrib || wrapped_distrib),
      D(wrapped_distrib || tfp_distrib).

    Args:
      method: the method name to be tested
    """
    try:
      expected_result1 = self.variant(
          getattr(self.tfp_second_dist, method))(self.base_distribution)
      expected_result2 = self.variant(
          getattr(self.base_distribution, method))(self.tfp_second_dist)
    except NotImplementedError:
      return
    except AttributeError:
      return
    distrax_result1 = self.variant(getattr(self.distrax_second_dist, method))(
        self.wrapped_dist)
    distrax_result2 = self.variant(getattr(self.wrapped_dist, method))(
        self.distrax_second_dist)
    tfp_result1 = self.variant(getattr(self.tfp_second_dist, method))(
        self.wrapped_dist)
    tfp_result2 = self.variant(getattr(self.wrapped_dist, method))(
        self.tfp_second_dist)
    self.assertion_fn(rtol=2e-4)(distrax_result1, expected_result1)
    self.assertion_fn(rtol=2e-4)(distrax_result2, expected_result2)
    self.assertion_fn(rtol=2e-4)(tfp_result1, expected_result1)
    self.assertion_fn(rtol=2e-4)(tfp_result2, expected_result2)


class DistributionFromTfpMvnNormal(DistributionFromTfpNormal):
  """Tests for multivariate normal distribution."""

  def setUp(self):
    super().setUp()
    self.base_dist = tfd.MultivariateNormalDiag(loc=[0., 1.])
    self.values = jnp.array([1., -1.])
    self.distrax_second_dist = MultivariateNormalDiag(
        loc=jnp.array([-1., 0.]), scale_diag=jnp.array([0.8, 1.2]))
    self.tfp_second_dist = tfd.MultivariateNormalDiag(
        loc=[-1., 0.], scale_diag=[0.8, 1.2])

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))))
  def test_slice(self, slice_):
    loc = np.random.randn(3, 4, 5)
    scale_diag = np.random.randn(3, 4, 5)
    dist = distribution_from_tfp(
        tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag))
    sliced_dist = dist[slice_]
    self.assertIsInstance(sliced_dist, Distribution)
    self.assertion_fn(rtol=2e-4)(sliced_dist.mean(), loc[slice_])


class DistributionFromTfpCategorical(DistributionFromTfpNormal):
  """Tests for categorical distribution."""

  def setUp(self):
    super().setUp()
    self.base_dist = tfd.Categorical(logits=[0., -1., 1.])
    self.values = jnp.array([0, 1, 2])
    self.distrax_second_dist = Categorical(probs=jnp.array([0.2, 0.2, 0.6]))
    self.tfp_second_dist = tfd.Categorical(probs=[0.2, 0.2, 0.6])

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
  )
  def test_slice(self, slice_):
    logits = np.random.randn(3, 4, 5)
    probs = jax.nn.softmax(np.random.randn(3, 4, 5), axis=-1)
    dist1 = distribution_from_tfp(tfd.Categorical(logits=logits))
    dist2 = distribution_from_tfp(tfd.Categorical(probs=probs))
    sliced_dist1 = dist1[slice_]
    sliced_dist2 = dist2[slice_]
    self.assertIsInstance(sliced_dist1, Distribution)
    self.assertIsInstance(sliced_dist2, Distribution)
    self.assertIsInstance(sliced_dist1, tfd.Categorical)
    self.assertIsInstance(sliced_dist2, tfd.Categorical)
    self.assertion_fn(rtol=2e-4)(
        jax.nn.softmax(sliced_dist1.logits, axis=-1),
        jax.nn.softmax(logits[slice_], axis=-1))
    self.assertion_fn(rtol=2e-4)(sliced_dist2.probs, probs[slice_])

  def test_slice_ellipsis(self):
    logits = np.random.randn(3, 4, 5)
    probs = jax.nn.softmax(np.random.randn(3, 4, 5), axis=-1)
    dist1 = distribution_from_tfp(tfd.Categorical(logits=logits))
    dist2 = distribution_from_tfp(tfd.Categorical(probs=probs))
    sliced_dist1 = dist1[..., -1]
    sliced_dist2 = dist2[..., -1]
    self.assertIsInstance(sliced_dist1, Distribution)
    self.assertIsInstance(sliced_dist2, Distribution)
    self.assertIsInstance(sliced_dist1, tfd.Categorical)
    self.assertIsInstance(sliced_dist2, tfd.Categorical)
    self.assertion_fn(rtol=2e-4)(
        jax.nn.softmax(sliced_dist1.logits, axis=-1),
        jax.nn.softmax(logits[:, -1], axis=-1))
    self.assertion_fn(rtol=2e-4)(sliced_dist2.probs, probs[:, -1])


class DistributionFromTfpTransformed(DistributionFromTfpNormal):
  """Tests for transformed distributions."""

  def setUp(self):
    super().setUp()
    self.base_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.Exp())
    self.values = jnp.array([0., 1., 2.])
    self.distrax_second_dist = Transformed(
        distribution=Normal(loc=0.5, scale=0.8),
        bijector=tfb.Exp())
    self.tfp_second_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0.5, scale=0.8),
        bijector=tfb.Exp())


if __name__ == '__main__':
  absltest.main()
