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
"""Tests for `mvn_diag.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import mvn_diag
from distrax._src.distributions import normal
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np


class MultivariateNormalDiagTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(mvn_diag.MultivariateNormalDiag)

  @parameterized.named_parameters(
      ('1d std normal', {'scale_diag': np.ones((1,))}),
      ('2d std normal', {'loc': [0., 0.], 'scale_diag': [1., 1.]}),
      ('2d std normal, None loc', {'scale_diag': [1., 1.]}),
      ('2d std normal, None scale_diag', {'loc': [0., 0.]}),
      ('rank-2 parameters',
       {'loc': np.zeros((3, 2)), 'scale_diag': np.ones((3, 2))}),
      ('broadcasted scale_diag',
       {'loc': np.zeros((3, 2)), 'scale_diag': np.ones((2,))}),
      ('broadcasted loc',
       {'loc': np.zeros((2)), 'scale_diag': np.ones((3, 2,))}),
  )
  def test_event_shape(self, distr_params):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    super()._test_event_shape((), distr_params)

  @parameterized.named_parameters(
      ('1d std normal', {'scale_diag': np.ones((1,))}),
      ('2d std normal', {'loc': [0., 0.], 'scale_diag': [1., 1.]}),
      ('2d std normal, None loc', {'scale_diag': [1., 1.]}),
      ('2d std normal, None scale_diag', {'loc': [0., 0.]}),
      ('rank-2 parameters',
       {'loc': np.zeros((3, 2)), 'scale_diag': np.ones((3, 2))}),
      ('broadcasted scale_diag',
       {'loc': np.zeros((3, 2)), 'scale_diag': np.ones((2,))}),
      ('broadcasted loc',
       {'loc': np.zeros((2)), 'scale_diag': np.ones((3, 2,))}),
  )
  def test_batch_shape(self, distr_params):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    super()._test_batch_shape((), distr_params)

  def test_invalid_parameters(self):
    self._test_raises_error(dist_kwargs={'loc': None, 'scale_diag': None})
    self._test_raises_error(
        dist_kwargs={'loc': None, 'scale_diag': np.array(1.)})
    self._test_raises_error(
        dist_kwargs={'loc': np.array(1.), 'scale_diag': None})
    self._test_raises_error(
        dist_kwargs={'loc': np.zeros((3, 5)), 'scale_diag': np.ones((3, 4))})

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std normal, no shape',
       {'scale_diag': np.ones((1,))},
       ()),
      ('2d std normal, no shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       ()),
      ('2d std normal, None loc, no shape',
       {'scale_diag': [1., 1.]},
       ()),
      ('2d std normal, None scale_diag, no shape',
       {'loc': [0., 0.]},
       ()),
      ('2d std normal, int shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       3),
      ('2d std normal, None loc, int shape',
       {'scale_diag': [1., 1.]},
       3),
      ('2d std normal, None scale_diag, int shape',
       {'loc': [0., 0.]},
       3),
      ('2d std normal, 1-tuple shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       (3,)),
      ('2d std normal, None loc, 1-tuple shape',
       {'scale_diag': [1., 1.]},
       (3,)),
      ('2d std normal, None scale_diag, 1-tuple shape',
       {'loc': [0., 0.]},
       (3,)),
      ('2d std normal, 2-tuple shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       (3, 4)),
      ('2d std normal, None loc, 2-tuple shape',
       {'scale_diag': [1., 1.]},
       (3, 4)),
      ('2d std normal, None scale_diag, 2-tuple shape',
       {'loc': [0., 0.]},
       (3, 4)),
      ('rank-2 parameters, 2-tuple shape',
       {'loc': np.zeros((3, 2)),
        'scale_diag': np.ones((3, 2))},
       (5, 4)),
      ('broadcasted scale_diag',
       {'loc': np.zeros((3, 2)),
        'scale_diag': np.ones((2,))},
       5),
      ('broadcasted loc',
       {'loc': np.zeros((2)),
        'scale_diag': np.ones((3, 2,))},
       5),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    super()._test_sample_shape(
        dist_args=(),
        dist_kwargs=distr_params,
        sample_shape=sample_shape)

  @chex.all_variants
  @jax.numpy_rank_promotion('raise')
  @parameterized.named_parameters(
      ('1d std normal, no shape',
       {'scale_diag': np.ones((1,))},
       ()),
      ('2d std normal, no shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       ()),
      ('2d std normal, None loc, no shape',
       {'scale_diag': [1., 1.]},
       ()),
      ('2d std normal, None scale_diag, no shape',
       {'loc': [0., 0.]},
       ()),
      ('2d std normal, int shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       3),
      ('2d std normal, None loc, int shape',
       {'scale_diag': [1., 1.]},
       3),
      ('2d std normal, None scale_diag, int shape',
       {'loc': [0., 0.]},
       3),
      ('2d std normal, 1-tuple shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       (3,)),
      ('2d std normal, None loc, 1-tuple shape',
       {'scale_diag': [1., 1.]},
       (3,)),
      ('2d std normal, None scale_diag, 1-tuple shape',
       {'loc': [0., 0.]},
       (3,)),
      ('2d std normal, 2-tuple shape',
       {'loc': [0., 0.],
        'scale_diag': [1., 1.]},
       (3, 4)),
      ('2d std normal, None loc, 2-tuple shape',
       {'scale_diag': [1., 1.]},
       (3, 4)),
      ('2d std normal, None scale_diag, 2-tuple shape',
       {'loc': [0., 0.]},
       (3, 4)),
      ('rank-2 parameters, 2-tuple shape',
       {'loc': np.zeros((3, 2)),
        'scale_diag': np.ones((3, 2))},
       (5, 4)),
      ('broadcasted scale_diag',
       {'loc': np.zeros((3, 2)),
        'scale_diag': np.ones((2,))},
       5),
      ('broadcasted loc',
       {'loc': np.zeros((2)),
        'scale_diag': np.ones((3, 2,))},
       5),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    super()._test_sample_and_log_prob(
        dist_args=(),
        dist_kwargs=distr_params,
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    dist_params = {
        'loc': np.array([0., 0.], dtype),
        'scale_diag': np.array([1., 1.], dtype)}
    dist = self.distrax_cls(**dist_params)
    samples = self.variant(dist.sample)(seed=self.key)
    self.assertEqual(samples.dtype, dist.dtype)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob; 2d dist, 2d value', 'log_prob',
       {'scale_diag': [1., 1.]},
       [0., -0.5]),
      ('log_prob; 3d dist, broadcasted params, 3d value', 'log_prob',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.3 * np.ones((3,))},
       [-0.1, 0., -0.5]),
      ('log_prob; 3d dist, broadcasted scale_diag, rank-2 value', 'log_prob',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((4, 3))),
      ('log_prob; 3d dist, broadcasted scale_diag, rank-3 value', 'log_prob',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((5, 4, 3))),
      ('log_prob; 2d dist, 2d value, edge case', 'log_prob',
       {'scale_diag': [1., 1.]},
       [200., -200.]),
      ('prob; 2d dist, 2d value', 'prob',
       {'scale_diag': [1., 1.]},
       [0., -0.5]),
      ('prob; 3d dist, broadcasted params, 3d value', 'prob',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.3 * np.ones((3,))},
       [-0.1, 0., -0.5]),
      ('prob; 3d dist, broadcasted scale_diag, rank-2 value', 'prob',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((4, 3))),
      ('prob; 3d dist, broadcasted scale_diag, rank-3 value', 'prob',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((5, 4, 3))),
      ('prob; 2d dist, 2d value, edge case', 'prob',
       {'scale_diag': [1., 1.]},
       [200., -200.]),
  )
  def test_pdf(self, function_string, distr_params, value):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    value = np.asarray(value)
    super()._test_attribute(
        attribute_string=function_string,
        dist_kwargs=distr_params,
        call_args=(value,),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_cdf; 2d dist, 2d value', 'log_cdf',
       {'scale_diag': [1., 1.]},
       [0., -0.5]),
      ('log_cdf; 3d dist, broadcasted params, 3d value', 'log_cdf',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.3 * np.ones((3,))},
       [-0.1, 0., -0.5]),
      ('log_cdf; 3d dist, broadcasted scale_diag, rank-2 value', 'log_cdf',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((4, 3))),
      ('log_cdf; 3d dist, broadcasted scale_diag, rank-3 value', 'log_cdf',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((5, 4, 3))),
      ('log_cdf; 2d dist, 2d value, edge case', 'log_cdf',
       {'scale_diag': [1., 1.]},
       [200., -200.]),
      ('cdf; 2d dist, 2d value', 'cdf',
       {'scale_diag': [1., 1.]},
       [0., -0.5]),
      ('cdf; 3d dist, broadcasted params, 3d value', 'cdf',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.3 * np.ones((3,))},
       [-0.1, 0., -0.5]),
      ('cdf; 3d dist, broadcasted scale_diag, rank-2 value', 'cdf',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((4, 3))),
      ('cdf; 3d dist, broadcasted scale_diag, rank-3 value', 'cdf',
       {'loc': np.zeros((4, 3)),
        'scale_diag': 0.1 * np.ones((3,))},
       -0.1 * np.ones((5, 4, 3))),
      ('cdf; 2d dist, 2d value, edge case', 'cdf',
       {'scale_diag': [1., 1.]},
       [200., -200.]),
  )
  def test_cdf(self, function_string, distr_params, value):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    value = np.asarray(value)
    dist = self.distrax_cls(**distr_params)
    result = self.variant(getattr(dist, function_string))(value)
    # The `cdf` is not implemented in TFP, so we test against a `Normal`.
    loc = 0. if 'loc' not in distr_params else distr_params['loc']
    univariate_normal = normal.Normal(loc, distr_params['scale_diag'])
    expected_result = getattr(univariate_normal, function_string)(value)
    if function_string == 'cdf':
      reduce_fn = lambda x: jnp.prod(x, axis=-1)
    elif function_string == 'log_cdf':
      reduce_fn = lambda x: jnp.sum(x, axis=-1)
    expected_result = reduce_fn(expected_result)
    self.assertion_fn(rtol=1e-3)(result, expected_result)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy; one distribution', 'entropy',
       {'loc': [0.1, -0.1],
        'scale_diag': [0.8, 0.5]}),
      ('entropy; broadcasted loc', 'entropy',
       {'loc': [0., 0.1, -0.1],
        'scale_diag': [[1.5, 0.8, 0.5], [0.8, 0.1, 0.4]]}),
      ('entropy; broadcasted scale_diag', 'entropy',
       {'loc': [[0., 0.1, -0.1], [0.2, 0., 0.5]],
        'scale_diag': [1.5, 0.8, 0.5]}),
      ('entropy; None loc', 'entropy',
       {'scale_diag': [0.8, 0.5]}),
      ('entropy; None scale_diag', 'entropy',
       {'loc': [0.1, -0.1]}),
      ('mean; one distribution', 'mean',
       {'loc': [0.1, -0.1],
        'scale_diag': [0.8, 0.5]}),
      ('mean; broadcasted loc', 'mean',
       {'loc': [0., 0.1, -0.1],
        'scale_diag': [[1.5, 0.8, 0.5], [0.8, 0.1, 0.4]]}),
      ('mean; broadcasted scale_diag', 'mean',
       {'loc': [[0., 0.1, -0.1], [0.2, 0., 0.5]],
        'scale_diag': [1.5, 0.8, 0.5]}),
      ('mean; None loc', 'mean',
       {'scale_diag': [0.8, 0.5]}),
      ('mean; None scale_diag', 'mean',
       {'loc': [0.1, -0.1]}),
      ('stddev; one distribution', 'stddev',
       {'loc': [0.1, -0.1],
        'scale_diag': [0.8, 0.5]}),
      ('stddev; broadcasted loc', 'stddev',
       {'loc': [0., 0.1, -0.1],
        'scale_diag': [[1.5, 0.8, 0.5], [0.8, 0.1, 0.4]]}),
      ('stddev; broadcasted scale_diag', 'stddev',
       {'loc': [[0., 0.1, -0.1], [0.2, 0., 0.5]],
        'scale_diag': [1.5, 0.8, 0.5]}),
      ('stddev; None loc', 'stddev',
       {'scale_diag': [0.8, 0.5]}),
      ('stddev; None scale_diag', 'stddev',
       {'loc': [0.1, -0.1]}),
      ('variance; one distribution', 'variance',
       {'loc': [0.1, -0.1],
        'scale_diag': [0.8, 0.5]}),
      ('variance; broadcasted loc', 'variance',
       {'loc': [0., 0.1, -0.1],
        'scale_diag': [[1.5, 0.8, 0.5], [0.8, 0.1, 0.4]]}),
      ('variance; broadcasted scale_diag', 'variance',
       {'loc': [[0., 0.1, -0.1], [0.2, 0., 0.5]],
        'scale_diag': [1.5, 0.8, 0.5]}),
      ('variance; None loc', 'variance',
       {'scale_diag': [0.8, 0.5]}),
      ('variance; None scale_diag', 'variance',
       {'loc': [0.1, -0.1]}),
      ('covariance; one distribution', 'covariance',
       {'loc': [0.1, -0.1],
        'scale_diag': [0.8, 0.5]}),
      ('covariance; broadcasted loc', 'covariance',
       {'loc': [0., 0.1, -0.1],
        'scale_diag': [[1.5, 0.8, 0.5], [0.8, 0.1, 0.4]]}),
      ('covariance; None loc', 'covariance',
       {'scale_diag': [0.8, 0.5]}),
      ('covariance; None scale_diag', 'covariance',
       {'loc': [0.1, -0.1]}),
      ('mode; broadcasted scale_diag', 'mode',
       {'loc': [[0., 0.1, -0.1], [0.2, 0., 0.5]],
        'scale_diag': [1.5, 0.8, 0.5]}),
  )
  def test_method(self, function_string, distr_params):
    distr_params = {
        k: np.asarray(v, dtype=np.float32) for k, v in distr_params.items()}
    super()._test_attribute(
        function_string,
        dist_kwargs=distr_params,
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @chex.all_variants(with_pmap=False)
  def test_median(self):
    dist_params = {'loc': np.array([0.3, -0.1, 0.0]),
                   'scale_diag': np.array([0.1, 1.4, 0.5])}
    dist = self.distrax_cls(**dist_params)
    self.assertion_fn(rtol=1e-3)(self.variant(dist.median)(), dist.mean())

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_two_distributions(self, function_string, mode_string):
    rng = np.random.default_rng(42)
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'loc': rng.normal(size=(4, 1, 5)).astype(np.float32),
            'scale_diag': 0.1 + rng.uniform(size=(3, 5)).astype(np.float32),
        },
        dist2_kwargs={
            'loc': np.asarray([-2.4, -1., 0., 1.2, 6.5]).astype(np.float32),
            'scale_diag': None,
        },
        assertion_fn=self.assertion_fn(rtol=1e-3))

  def test_jittable(self):
    super()._test_jittable(
        (np.zeros((2, 3,)), np.ones((2, 3,))),
        assertion_fn=self.assertion_fn(rtol=1e-3))

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
  )
  def test_slice(self, slice_):
    rng = np.random.default_rng(42)
    loc = jnp.array(rng.normal(size=(3, 4, 5)))
    scale_diag = jnp.array(rng.uniform(size=(3, 4, 5)))
    dist = self.distrax_cls(loc=loc, scale_diag=scale_diag)
    self.assertion_fn(rtol=1e-3)(dist[slice_].mean(), loc[slice_])

  def test_slice_different_parameterization(self):
    rng = np.random.default_rng(42)
    loc = jnp.array(rng.normal(size=(4,)))
    scale_diag = jnp.array(rng.uniform(size=(3, 4)))
    dist = self.distrax_cls(loc=loc, scale_diag=scale_diag)
    self.assertion_fn(rtol=1e-3)(dist[0].mean(), loc)  # Not slicing loc.
    self.assertion_fn(rtol=1e-3)(dist[0].stddev(), scale_diag[0])

  def test_slice_ellipsis(self):
    rng = np.random.default_rng(42)
    loc = jnp.array(rng.normal(size=(3, 4, 5)))
    scale_diag = jnp.array(rng.uniform(size=(3, 4, 5)))
    dist = self.distrax_cls(loc=loc, scale_diag=scale_diag)
    self.assertion_fn(rtol=1e-3)(dist[..., -1].mean(), loc[:, -1])
    self.assertion_fn(rtol=1e-3)(dist[..., -1].stddev(), scale_diag[:, -1])


if __name__ == '__main__':
  absltest.main()
