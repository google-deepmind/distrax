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
"""Tests for `tfp_compatible_distribution.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.independent import Independent
from distrax._src.distributions.laplace import Laplace
from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.normal import Normal
from distrax._src.distributions.tfp_compatible_distribution import tfp_compatible_distribution
from distrax._src.distributions.transformed import Transformed
from distrax._src.distributions.uniform import Uniform
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class TFPCompatibleDistributionNormal(parameterized.TestCase):
  """Tests for Normal distribution."""

  def setUp(self):
    super().setUp()
    self._sample_shape = (np.int32(10),)
    self._seed = 42
    self._key = jax.random.PRNGKey(self._seed)
    self.base_dist = Normal(loc=jnp.array([0., 0.]), scale=jnp.array([1., 1.]))
    self.values = jnp.array([1., -1.])
    self.distrax_second_dist = Normal(loc=-1., scale=0.8)
    self.tfp_second_dist = tfd.Normal(loc=-1., scale=0.8)

  def assertion_fn(self, rtol):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  @property
  def wrapped_dist(self):
    return tfp_compatible_distribution(self.base_dist)

  def test_event_shape(self):
    self.assertEqual(self.wrapped_dist.event_shape, self.base_dist.event_shape)

  def test_event_shape_types(self):
    wrapped_dist = tfp_compatible_distribution(self.distrax_second_dist)
    self.assertEqual(
        type(wrapped_dist.event_shape), type(self.tfp_second_dist.event_shape))
    self.assertEqual(
        type(wrapped_dist.event_shape_tensor()),
        type(self.tfp_second_dist.event_shape_tensor()))

  def test_batch_shape(self):
    self.assertEqual(self.wrapped_dist.batch_shape, self.base_dist.batch_shape)

  @chex.all_variants
  def test_sample(self):
    def sample_fn(key):
      return self.wrapped_dist.sample(seed=key, sample_shape=self._sample_shape)
    sample_fn = self.variant(sample_fn)
    self.assertion_fn(rtol=1e-4)(
        sample_fn(self._key),
        self.base_dist.sample(sample_shape=self._sample_shape, seed=self._key))

  def test_experimental_local_measure(self):
    samples = self.wrapped_dist.sample(seed=self._key)
    expected_log_prob = self.wrapped_dist.log_prob(samples)

    log_prob, space = self.wrapped_dist.experimental_local_measure(
        samples, backward_compat=True)
    self.assertion_fn(rtol=1e-4)(log_prob, expected_log_prob)
    self.assertIsInstance(space, tfp.experimental.tangent_spaces.FullSpace)

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
    try:
      expected_result = self.variant(getattr(self.base_dist, method))()
    except NotImplementedError:
      return
    except AttributeError:
      return
    result = self.variant(getattr(self.wrapped_dist, method))()
    self.assertion_fn(rtol=1e-4)(result, expected_result)

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob', 'log_prob'),
      ('prob', 'prob'),
      ('log_cdf', 'log_cdf'),
      ('cdf', 'cdf'),
  )
  def test_method_with_value(self, method):
    try:
      expected_result = self.variant(
          getattr(self.base_dist, method))(self.values)
    except NotImplementedError:
      return
    except AttributeError:
      return
    result = self.variant(getattr(self.wrapped_dist, method))(self.values)
    self.assertion_fn(rtol=1e-4)(result, expected_result)

  @chex.all_variants
  @parameterized.named_parameters(
      ('kl_divergence', 'kl_divergence'),
      ('cross_entropy', 'cross_entropy'),
  )
  def test_with_two_distributions(self, method):
    """Test methods of the form listed below.

      D(distrax_distrib || wrapped_distrib),
      D(wrapped_distrib || distrax_distrib),
      D(tfp_distrib || wrapped_distrib),
      D(wrapped_distrib || tfp_distrib).

    Args:
      method: the method name to be tested
    """
    try:
      expected_result1 = self.variant(
          getattr(self.distrax_second_dist, method))(self.base_distribution)
      expected_result2 = self.variant(
          getattr(self.base_distribution, method))(self.distrax_second_dist)
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
    self.assertion_fn(rtol=1e-4)(distrax_result1, expected_result1)
    self.assertion_fn(rtol=1e-4)(distrax_result2, expected_result2)
    self.assertion_fn(rtol=1e-4)(tfp_result1, expected_result1)
    self.assertion_fn(rtol=1e-4)(tfp_result2, expected_result2)


class TFPCompatibleDistributionMvnNormal(TFPCompatibleDistributionNormal):
  """Tests for multivariate normal distribution."""

  def setUp(self):
    super().setUp()
    self.base_dist = MultivariateNormalDiag(loc=jnp.array([0., 1.]))
    self.values = jnp.array([1., -1.])
    self.distrax_second_dist = MultivariateNormalDiag(
        loc=jnp.array([-1., 0.]), scale_diag=jnp.array([0.8, 1.2]))
    self.tfp_second_dist = tfd.MultivariateNormalDiag(
        loc=jnp.array([-1., 0.]), scale_diag=jnp.array([0.8, 1.2]))


class TFPCompatibleDistributionCategorical(TFPCompatibleDistributionNormal):
  """Tests for categorical distribution."""

  def setUp(self):
    super().setUp()
    self.base_dist = Categorical(logits=jnp.array([0., -1., 1.]))
    self.values = jnp.array([0, 1, 2])
    self.distrax_second_dist = Categorical(probs=jnp.array([0.2, 0.2, 0.6]))
    self.tfp_second_dist = tfd.Categorical(probs=jnp.array([0.2, 0.2, 0.6]))


class TFPCompatibleDistributionTransformed(TFPCompatibleDistributionNormal):
  """Tests for transformed distributions."""

  def setUp(self):
    super().setUp()
    self.base_dist = Transformed(
        distribution=Normal(loc=0., scale=1.),
        bijector=tfb.Exp())
    self.values = jnp.array([0., 1., 2.])
    self.distrax_second_dist = Transformed(
        distribution=Normal(loc=0.5, scale=0.8),
        bijector=tfb.Exp())
    self.tfp_second_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0.5, scale=0.8),
        bijector=tfb.Exp())


class TfpMetaDistributionsWithWrappedBaseDistribution(parameterized.TestCase):
  """Tests for meta distributions (with wrappper base distr)."""

  def setUp(self):
    super().setUp()
    self._sample_shape = (np.int32(10),)
    self._seed = 42
    self._key = jax.random.PRNGKey(self._seed)

  def assertion_fn(self, rtol):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  def test_with_independent(self):
    base_dist = Normal(loc=jnp.array([0., 0.]), scale=jnp.array([1., 1.]))
    wrapped_dist = tfp_compatible_distribution(base_dist)

    meta_dist = tfd.Independent(wrapped_dist, 1, validate_args=True)
    samples = meta_dist.sample((), self._key)
    log_prob = meta_dist.log_prob(samples)

    distrax_meta_dist = Independent(base_dist, 1)
    expected_log_prob = distrax_meta_dist.log_prob(samples)

    self.assertion_fn(rtol=1e-4)(log_prob, expected_log_prob)

  def test_with_transformed_distribution(self):
    base_dist = Normal(loc=jnp.array([0., 0.]), scale=jnp.array([1., 1.]))
    wrapped_dist = tfp_compatible_distribution(base_dist)

    meta_dist = tfd.TransformedDistribution(
        distribution=wrapped_dist, bijector=tfb.Exp(), validate_args=True)
    samples = meta_dist.sample(seed=self._key)
    log_prob = meta_dist.log_prob(samples)

    distrax_meta_dist = Transformed(
        distribution=base_dist, bijector=tfb.Exp())
    expected_log_prob = distrax_meta_dist.log_prob(samples)

    self.assertion_fn(rtol=1e-4)(log_prob, expected_log_prob)

  def test_with_sample(self):
    base_dist = Normal(0., 1.)
    wrapped_dist = tfp_compatible_distribution(base_dist)
    meta_dist = tfd.Sample(
        wrapped_dist, sample_shape=[1, 3], validate_args=True)
    meta_dist.log_prob(meta_dist.sample(2, seed=self._key))

  def test_with_joint_distribution_named_auto_batched(self):
    def laplace(a, b):
      return tfp_compatible_distribution(Laplace(a * jnp.ones((2, 1)), b))
    meta_dist = tfd.JointDistributionNamedAutoBatched({
        'a': tfp_compatible_distribution(Uniform(2. * jnp.ones(3), 4.)),
        'b': tfp_compatible_distribution(Uniform(2. * jnp.ones(3), 4.)),
        'c': laplace}, validate_args=True)
    meta_dist.log_prob(meta_dist.sample(4, seed=self._key))

  def test_with_joint_distribution_coroutine_auto_batched(self):

    def model_fn():
      a = yield tfp_compatible_distribution(Uniform(2. * jnp.ones(3), 4.),
                                            name='a')
      b = yield tfp_compatible_distribution(Uniform(2. * jnp.ones(3), 4.),
                                            name='b')
      yield tfp_compatible_distribution(Laplace(a * jnp.ones((2, 1)), b),
                                        name='c')

    meta_dist = tfd.JointDistributionCoroutineAutoBatched(
        model_fn, validate_args=True)
    meta_dist.log_prob(meta_dist.sample(7, seed=self._key))


class TFPCompatibleDistributionSlicing(parameterized.TestCase):
  """Class to test the `getitem` method."""

  def assertion_fn(self, rtol):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    loc = np.random.randn(3, 4, 5)
    base_dist = Normal(loc=loc, scale=1.)
    dist = tfp_compatible_distribution(base_dist)
    sliced_dist = dist[slice_]
    self.assertIsInstance(sliced_dist, base_dist.__class__)
    self.assertIsInstance(sliced_dist.batch_shape, tfp.tf2jax.TensorShape)
    self.assertTrue(sliced_dist.allow_nan_stats)
    self.assertion_fn(rtol=1e-4)(sliced_dist.loc, loc[slice_])


if __name__ == '__main__':
  absltest.main()
