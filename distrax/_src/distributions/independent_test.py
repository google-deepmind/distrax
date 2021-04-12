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
"""Tests for `independent.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import independent
from distrax._src.distributions import mvn_diag
from distrax._src.distributions import normal
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

RTOL = 1e-3


class IndependentTest(parameterized.TestCase):

  @parameterized.parameters(None, 0, 1, 2)
  def test_constructor_is_jittable_given_ndims(self, ndims):
    base = normal.Normal(loc=jnp.zeros((2, 3)), scale=jnp.ones((2, 3)))
    constructor = lambda d: independent.Independent(d, ndims)
    jax.jit(constructor)(base)


class TFPMultivariateNormalTest(equivalence.EquivalenceTest,
                                parameterized.TestCase):
  """Class to test Distrax Independent distribution against its TFP counterpart.

  This class tests the case when using a TFP multivariate Normal distribution
  as input for the TFP and Distrax Independent. There are 2 methods to create
  the base distributions, `_make_base_distribution` and
  `_make_tfp_base_distribution`. By overloading these methods, different
  base distributions can be used.
  """

  def _make_tfp_base_distribution(self, loc, scale):
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

  # Define the function to create the base distribution.
  _make_base_distribution = _make_tfp_base_distribution

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(independent.Independent)

    self.normal_loc = jax.random.normal(
        key=jax.random.PRNGKey(42), shape=(5, 4, 3, 2))
    self.normal_scale = 0.5 + np.abs(
        jax.random.normal(key=jax.random.PRNGKey(43), shape=(5, 4, 3, 2)))
    self.normal_loc2 = jax.random.normal(
        key=jax.random.PRNGKey(43), shape=(5, 4, 3, 2))
    self.normal_scale2 = 0.5 + np.abs(
        jax.random.normal(key=jax.random.PRNGKey(44), shape=(5, 4, 3, 2)))

    # For most tests, we use `base_dist` and `tfp_base_dist` as base
    # distributions. The latter is used as input for the TFP Independent, which
    # we compare against.
    self.base_dist = self._make_base_distribution(self.normal_loc,
                                                  self.normal_scale)
    self.tfp_base_dist = self._make_tfp_base_distribution(self.normal_loc,
                                                          self.normal_scale)
    # Some methods (e.g., the KL divergence) require two distributions. We
    # define here the base distribution for those methods.
    self.base_dist2 = self._make_base_distribution(self.normal_loc2,
                                                   self.normal_scale2)
    self.tfp_base_dist2 = self._make_tfp_base_distribution(self.normal_loc2,
                                                           self.normal_scale2)

    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  def test_invalid_parameters(self):
    self._test_raises_error(
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': -1})
    self._test_raises_error(
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': 10})

  @parameterized.named_parameters(
      ('batch dims None', None),
      ('batch dims 0', 0),
      ('batch dims 1', 1),
      ('batch dims 2', 2),
      ('batch dims 3', 3),
  )
  def test_event_shape(self, batch_ndims):
    super()._test_event_shape(
        (),
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist_kwargs={'distribution': self.tfp_base_dist,
                         'reinterpreted_batch_ndims': batch_ndims},
    )

  @parameterized.named_parameters(
      ('batch dims None', None),
      ('batch dims 0', 0),
      ('batch dims 1', 1),
      ('batch dims 2', 2),
      ('batch dims 3', 3),
  )
  def test_batch_shape(self, batch_ndims):
    super()._test_batch_shape(
        (),
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist_kwargs={'distribution': self.tfp_base_dist,
                         'reinterpreted_batch_ndims': batch_ndims},
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('batch dims None, empty shape', None, ()),
      ('batch dims None, int shape', None, 10),
      ('batch dims None, 2-tuple shape', None, (10, 20)),
      ('batch dims 1, empty shape', 1, ()),
      ('batch dims 1, int shape', 1, 10),
      ('batch dims 1, 2-tuple shape', 1, (10, 20)),
      ('batch dims 3, empty shape', 3, ()),
      ('batch dims 3, int shape', 3, 10),
      ('batch dims 3, 2-tuple shape', 3, (10, 20)),
  )
  def test_sample_shape(self, batch_ndims, sample_shape):
    super()._test_sample_shape(
        (),
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist_kwargs={'distribution': self.tfp_base_dist,
                         'reinterpreted_batch_ndims': batch_ndims},
        sample_shape=sample_shape)

  @chex.all_variants
  def test_sample_dtype(self):
    dist = self.distrax_cls(self.base_dist)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(dist.dtype, samples.dtype)
    self.assertEqual(dist.dtype, self.base_dist.dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('batch dims None, empty shape', None, ()),
      ('batch dims None, int shape', None, 10),
      ('batch dims None, 2-tuple shape', None, (10, 20)),
      ('batch dims 1, empty shape', 1, ()),
      ('batch dims 1, int shape', 1, 10),
      ('batch dims 1, 2-tuple shape', 1, (10, 20)),
      ('batch dims 3, empty shape', 3, ()),
      ('batch dims 3, int shape', 3, 10),
      ('batch dims 3, 2-tuple shape', 3, (10, 20)),
  )
  def test_sample_and_log_prob(self, batch_ndims, sample_shape):
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist_kwargs={'distribution': self.tfp_base_dist,
                         'reinterpreted_batch_ndims': batch_ndims},
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn)

  @chex.all_variants
  @parameterized.named_parameters(
      ('batch dims None', None, np.zeros((5, 4, 3, 2))),
      ('batch dims 0', 0, np.zeros((5, 4, 3, 2))),
      ('batch dims 1', 1, np.zeros((5, 4, 3, 2))),
      ('batch dims 2', 2, np.zeros((5, 4, 3, 2))),
      ('batch dims 3', 3, np.zeros((5, 4, 3, 2))),
  )
  def test_log_prob(self, batch_ndims, value):
    super()._test_attribute(
        attribute_string='log_prob',
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist_kwargs={'distribution': self.tfp_base_dist,
                         'reinterpreted_batch_ndims': batch_ndims},
        call_args=(value,),
        assertion_fn=self.assertion_fn)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy, batch dims None', 'entropy', None),
      ('entropy, batch dims 1', 'entropy', 1),
      ('entropy, batch dims 3', 'entropy', 3),
      ('mean, batch dims None', 'mean', None),
      ('mean, batch dims 1', 'mean', 1),
      ('mean, batch dims 3', 'mean', 3),
      ('variance, batch dims None', 'variance', None),
      ('variance, batch dims 1', 'variance', 1),
      ('variance, batch dims 3', 'variance', 3),
      ('stddev, batch dims None', 'stddev', None),
      ('stddev, batch dims 1', 'stddev', 1),
      ('stddev, batch dims 3', 'stddev', 3),
      ('mode, batch dims None', 'mode', None),
      ('mode, batch dims 1', 'mode', 1),
      ('mode, batch dims 3', 'mode', 3),
  )
  def test_method(self, function_string, batch_ndims):
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist_kwargs={'distribution': self.tfp_base_dist,
                         'reinterpreted_batch_ndims': batch_ndims},
        assertion_fn=self.assertion_fn)

  @chex.all_variants(with_jit=False, with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax, no batch dims',
       'kl_divergence', 'distrax_to_distrax', None),
      ('kl distrax_to_tfp, no batch dims',
       'kl_divergence', 'distrax_to_tfp', None),
      ('kl tfp_to_distrax, no batch dims',
       'kl_divergence', 'tfp_to_distrax', None),
      ('cross-ent distrax_to_distrax, no batch dims',
       'cross_entropy', 'distrax_to_distrax', None),
      ('cross-ent distrax_to_tfp, no batch dims',
       'cross_entropy', 'distrax_to_tfp', None),
      ('cross-ent tfp_to_distrax, no batch dims',
       'cross_entropy', 'tfp_to_distrax', None),
      ('kl distrax_to_distrax, batch dims 2',
       'kl_divergence', 'distrax_to_distrax', 2),
      ('kl distrax_to_tfp, batch dims 2',
       'kl_divergence', 'distrax_to_tfp', 2),
      ('kl tfp_to_distrax, batch dims 2',
       'kl_divergence', 'tfp_to_distrax', 2),
      ('cross-ent distrax_to_distrax, batch dims 2',
       'cross_entropy', 'distrax_to_distrax', 2),
      ('cross-ent distrax_to_tfp, batch dims 2',
       'cross_entropy', 'distrax_to_tfp', 2),
      ('cross-ent tfp_to_distrax, batch dims 2',
       'cross_entropy', 'tfp_to_distrax', 2),
  )
  def test_with_two_distributions(self, function_string, mode_string,
                                  batch_ndims):
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={'distribution': self.base_dist,
                      'reinterpreted_batch_ndims': batch_ndims},
        dist2_kwargs={'distribution': self.base_dist2,
                      'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist1_kwargs={'distribution': self.tfp_base_dist,
                          'reinterpreted_batch_ndims': batch_ndims},
        tfp_dist2_kwargs={'distribution': self.tfp_base_dist2,
                          'reinterpreted_batch_ndims': batch_ndims},
        assertion_fn=self.assertion_fn)


class TFPUnivariateNormalTest(TFPMultivariateNormalTest):
  """Class to test Distrax Independent distribution against its TFP counterpart.

  This class tests the case when using a TFP univariate Normal distribution
  as input for the TFP and distrax Independent.
  """

  def _make_tfp_base_distribution(self, loc, scale):
    return tfd.Normal(loc=loc, scale=scale)

  # Define the function to create the base distribution.
  _make_base_distribution = _make_tfp_base_distribution

  def test_jittable(self):
    super()._test_jittable(
        dist_kwargs={'distribution': self.base_dist,
                     'reinterpreted_batch_ndims': 1},
        assertion_fn=functools.partial(np.testing.assert_allclose, rtol=1e-4))


class DistraxUnivariateNormalTest(TFPMultivariateNormalTest):
  """Class to test Distrax Independent distribution against its TFP counterpart.

  This class tests the case when using a distrax univariate Normal distribution
  as input for the Distrax Independent.
  """

  def _make_distrax_base_distribution(self, loc, scale):
    return normal.Normal(loc=loc, scale=scale)

  def _make_tfp_base_distribution(self, loc, scale):
    return tfd.Normal(loc=loc, scale=scale)

  # Define the function to create the base distribution.
  _make_base_distribution = _make_distrax_base_distribution


class DistraxMultivariateNormalTest(TFPMultivariateNormalTest):
  """Class to test Distrax Independent distribution against its TFP counterpart.

  This class tests the case when using a Distrax multivariate Normal
  distribution as input for the Distrax Independent.
  """

  def _make_distrax_base_distribution(self, loc, scale):
    return mvn_diag.MultivariateNormalDiag(loc=loc, scale_diag=scale)

  # Define the function to create the base distribution.
  _make_base_distribution = _make_distrax_base_distribution


if __name__ == '__main__':
  absltest.main()
