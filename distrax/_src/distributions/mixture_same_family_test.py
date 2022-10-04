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
"""Tests for `mixture_same_family.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.mixture_same_family import MixtureSameFamily
from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.normal import Normal
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class TFPLogitsCategoricalTFPMultivariateComponents(equivalence.EquivalenceTest
                                                   ):
  """Class to test distrax mixture against tfp mixture.

  There are 4 methods to create categorical and components distributions (one
  each for tfp and distrax). Those are used to instantiate the input
  distributions for the mixtures to be tested against each other. By overloading
  these methods, different combinations can be tested.

  This class tests the case when using tfp distributions (categorical from
  logits) as inputs in the tfp and distrax mixture.
  """

  def _make_tfp_categorical(self, logits, probs):
    """Categorical distribution for tfp reference mixture."""
    del probs
    return tfd.Categorical(logits=logits)

  def _make_tfp_components(self, key_loc, key_scale):
    """Components distribution for tfp reference mixture."""
    components_shape = self.batch_shape + (self.num_components,) + (2,)
    return tfd.MultivariateNormalDiag(
        loc=jax.random.normal(key=key_loc, shape=components_shape),
        scale_diag=jax.random.uniform(key=key_scale, shape=components_shape)+.5)

  # Define functions to create input distributions for the Distrax mixture. This
  # class tests Distrax mixture using the same TFP input distributions in both
  # TFP and Distrax. Subclasses will use different combinations.
  _make_categorical = _make_tfp_categorical
  _make_components = _make_tfp_components

  def setUp(self):
    super().setUp()
    self._init_distr_cls(MixtureSameFamily)

    self.batch_shape = (5, 4)
    self.num_components = 3

    logits_shape = self.batch_shape + (self.num_components,)
    logits = jax.random.normal(key=jax.random.PRNGKey(42),
                               shape=logits_shape)
    probs = jax.nn.softmax(logits, axis=-1)
    self.mixture_dist = self._make_categorical(logits, probs)
    self.tfp_mixture_dist = self._make_tfp_categorical(logits, probs)

    key_loc, key_scale = jax.random.split(jax.random.PRNGKey(42))
    self.components_dist = self._make_components(key_loc, key_scale)
    self.tfp_components_dist = self._make_tfp_components(key_loc, key_scale)

  def test_event_shape(self):
    super()._test_event_shape(
        (),
        dist_kwargs={
            'mixture_distribution': self.mixture_dist,
            'components_distribution': self.components_dist
        },
        tfp_dist_kwargs={
            'mixture_distribution': self.tfp_mixture_dist,
            'components_distribution': self.tfp_components_dist
        },
    )

  def test_batch_shape(self):
    super()._test_batch_shape(
        (),
        dist_kwargs={
            'mixture_distribution': self.mixture_dist,
            'components_distribution': self.components_dist
        },
        tfp_dist_kwargs={
            'mixture_distribution': self.tfp_mixture_dist,
            'components_distribution': self.tfp_components_dist
        },
    )

  def test_invalid_parameters(self):
    logits_shape = (1,) + self.batch_shape + (self.num_components,)
    logits = jnp.ones(logits_shape, dtype=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    key_loc, key_scale = jax.random.split(jax.random.PRNGKey(42))
    self._test_raises_error(dist_kwargs={
        'mixture_distribution': self._make_categorical(logits, probs),
        'components_distribution': self._make_components(key_loc, key_scale),
    })

  @chex.all_variants
  @parameterized.named_parameters(
      ('empty shape', ()),
      ('int shape', 10),
      ('2-tuple shape', (10, 20)),
  )
  def test_sample_shape(self, sample_shape):
    super()._test_sample_shape(
        (),
        dist_kwargs={
            'mixture_distribution': self.mixture_dist,
            'components_distribution': self.components_dist
        },
        tfp_dist_kwargs={
            'mixture_distribution': self.tfp_mixture_dist,
            'components_distribution': self.tfp_components_dist
        },
        sample_shape=sample_shape)

  @chex.all_variants
  def test_sample_dtype(self):
    dist = self.distrax_cls(
        mixture_distribution=self.mixture_dist,
        components_distribution=self.components_dist)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(dist.dtype, samples.dtype)
    self.assertEqual(dist.dtype, self.components_dist.dtype)

  @chex.all_variants()
  @parameterized.named_parameters(
      ('empty shape', ()),
      ('int shape', 10),
      ('2-tuple shape', (10, 20)),
  )
  def test_sample_and_log_prob(self, sample_shape):
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs={
            'mixture_distribution': self.mixture_dist,
            'components_distribution': self.components_dist
        },
        tfp_dist_kwargs={
            'mixture_distribution': self.tfp_mixture_dist,
            'components_distribution': self.tfp_components_dist
        },
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=2e-3))

  # `pmap` must have at least one non-None value in `in_axes`.
  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('mean', 'mean'),
      ('variance', 'variance'),
      ('stddev', 'stddev'),
  )
  def test_method(self, function_string):
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs={
            'mixture_distribution': self.mixture_dist,
            'components_distribution': self.components_dist
        },
        tfp_dist_kwargs={
            'mixture_distribution': self.tfp_mixture_dist,
            'components_distribution': self.tfp_components_dist
        },
        assertion_fn=self.assertion_fn(rtol=2e-3))

  def test_jittable(self):
    super()._test_jittable(
        dist_kwargs={
            'mixture_distribution': self.mixture_dist,
            'components_distribution': self.components_dist},
        assertion_fn=self.assertion_fn(rtol=1e-3))


class TFPLogitsCategoricalTFPUnivariateComponents(
    TFPLogitsCategoricalTFPMultivariateComponents):

  def _make_tfp_components(self, key_loc, key_scale):
    components_shape = self.batch_shape + (self.num_components,)
    return tfd.Normal(
        loc=jax.random.normal(key=key_loc, shape=components_shape),
        scale=jax.random.uniform(key=key_scale, shape=components_shape)+0.5)

  _make_components = _make_tfp_components


# Enough to only test one type of components for `tfp.Categorical` with `probs`.
class TFPProbsCategoricalTFPMultivariateComponents(
    TFPLogitsCategoricalTFPMultivariateComponents):

  def _make_categorical(self, logits, probs):
    del logits
    return tfd.Categorical(probs=probs)


class DistraxLogitsCategoricalTFPMultivariateComponents(
    TFPLogitsCategoricalTFPMultivariateComponents):

  def _make_categorical(self, logits, probs):
    del probs
    return Categorical(logits=logits)


class DistraxProbsCategoricalTFPMultivariateComponents(
    TFPLogitsCategoricalTFPMultivariateComponents):

  def _make_categorical(self, logits, probs):
    del logits
    return Categorical(probs=probs)


class DistraxLogitsCategoricalTFPUnivariateComponents(
    TFPLogitsCategoricalTFPUnivariateComponents):

  def _make_categorical(self, logits, probs):
    del probs
    return Categorical(logits=logits)


class DistraxLogitsCategoricalDistraxMultivariateComponents(
    DistraxLogitsCategoricalTFPMultivariateComponents):

  def _make_components(self, key_loc, key_scale):
    components_shape = self.batch_shape + (self.num_components,) + (2,)
    return MultivariateNormalDiag(
        loc=jax.random.normal(key=key_loc, shape=components_shape),
        scale_diag=jax.random.uniform(key=key_scale, shape=components_shape) +
        0.5)


class DistraxLogitsCategoricalDistraxUnivariateComponents(
    DistraxLogitsCategoricalTFPUnivariateComponents):

  def _make_components(self, key_loc, key_scale):
    components_shape = self.batch_shape + (self.num_components,)
    return Normal(
        loc=jax.random.normal(key=key_loc, shape=components_shape),
        scale=jax.random.uniform(key=key_scale, shape=components_shape) + 0.5)


class TFPLogitsCategoricalDistraxMultivariateComponents(
    TFPLogitsCategoricalTFPMultivariateComponents):

  def _make_components(self, key_loc, key_scale):
    components_shape = self.batch_shape + (self.num_components,) + (2,)
    return MultivariateNormalDiag(
        loc=jax.random.normal(key=key_loc, shape=components_shape),
        scale_diag=jax.random.uniform(key=key_scale, shape=components_shape) +
        0.5)


class TFPLogitsCategoricalDistraxUnivariateComponents(
    TFPLogitsCategoricalTFPUnivariateComponents):

  def _make_components(self, key_loc, key_scale):
    components_shape = self.batch_shape + (self.num_components,)
    return Normal(
        loc=jax.random.normal(key=key_loc, shape=components_shape),
        scale=jax.random.uniform(key=key_scale, shape=components_shape) + 0.5)


class MixtureSameFamilySlicingTest(parameterized.TestCase):
  """Class to test the `getitem` method."""

  def setUp(self):
    super().setUp()
    self.loc = np.random.randn(2, 3, 4, 5)
    self.scale_diag = np.abs(np.random.randn(2, 3, 4, 5))
    self.components_dist = MultivariateNormalDiag(
        loc=self.loc, scale_diag=self.scale_diag)
    self.logits = np.random.randn(2, 3, 4)
    self.mixture_dist = Categorical(logits=self.logits)
    self.dist = MixtureSameFamily(self.mixture_dist, self.components_dist)

  def assertion_fn(self, rtol):
    return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

  @parameterized.named_parameters(
      ('single element', 1, (3,)),
      ('range', slice(-1), (1, 3)),
      ('range_2', (slice(None), slice(-1)), (2, 2)),
  )
  def test_slice(self, slice_, expected_batch_shape):
    sliced_dist = self.dist[slice_]
    self.assertEqual(sliced_dist.batch_shape, expected_batch_shape)
    self.assertEqual(sliced_dist.event_shape, self.dist.event_shape)
    self.assertEqual(sliced_dist.mixture_distribution.logits.shape[-1],
                     self.dist.mixture_distribution.logits.shape[-1])
    self.assertIsInstance(sliced_dist, MixtureSameFamily)
    self.assertIsInstance(
        sliced_dist.components_distribution, MultivariateNormalDiag)
    self.assertIsInstance(sliced_dist.mixture_distribution, Categorical)
    self.assertion_fn(rtol=2e-3)(
        sliced_dist.components_distribution.loc, self.loc[slice_])
    self.assertion_fn(rtol=2e-3)(
        sliced_dist.components_distribution.scale_diag, self.scale_diag[slice_])

  def test_slice_ellipsis(self):
    sliced_dist = self.dist[..., -1]
    expected_batch_shape = (2,)
    self.assertEqual(sliced_dist.batch_shape, expected_batch_shape)
    self.assertEqual(sliced_dist.event_shape, self.dist.event_shape)
    self.assertEqual(sliced_dist.mixture_distribution.logits.shape[-1],
                     self.dist.mixture_distribution.logits.shape[-1])
    self.assertIsInstance(sliced_dist, MixtureSameFamily)
    self.assertIsInstance(
        sliced_dist.components_distribution, MultivariateNormalDiag)
    self.assertIsInstance(sliced_dist.mixture_distribution, Categorical)
    self.assertion_fn(rtol=2e-3)(
        sliced_dist.components_distribution.loc,
        self.loc[:, -1, ...])
    self.assertion_fn(rtol=2e-3)(
        sliced_dist.components_distribution.scale_diag,
        self.scale_diag[:, -1, ...])


if __name__ == '__main__':
  absltest.main()
