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
"""Tests for `mvn_linear_operator.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import mvn_linear_operator
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


tfb = tfp.bijectors

RTOL = 1e-3


class MultivariateNormalLinearOperatorTest(
    equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(mvn_linear_operator.MultivariateNormalLinearOperator)
    self.key, self.key_bijector, self.key_val = jax.random.split(
        self.key, num=3)
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  def _create_random_bijector(self, loc_shape, scale_type, scale_shape,
                              dtype=jnp.float32, key=None):
    key = self.key_bijector if key is None else key
    key_loc, key_scale = jax.random.split(key, num=2)
    loc = jax.random.normal(key_loc, shape=loc_shape).astype(dtype)
    if scale_type == 'scale_tril':
      scale = {
          'scale_tril': jnp.vectorize(
              jax.scipy.linalg.tril, signature='(k,k)->(k,k)')(
                  jax.random.normal(key_scale, scale_shape)).astype(dtype)
      }
    elif scale_type == 'scale_diag':
      scale = {
          'scale_diag': jax.random.normal(
              key_scale, scale_shape).astype(dtype)
      }
    else:
      raise ValueError('Unknown `scale_type`.')
    return tfb.Affine(shift=loc, **scale)

  @parameterized.named_parameters(
      ('1d loc, 2d scale_tril', (4,), 'scale_tril', (4, 4)),
      ('1d loc, 3d scale_tril', (4,), 'scale_tril', (3, 4, 4)),
      ('2d loc, 2d scale_tril', (3, 4), 'scale_tril', (4, 4)),
      ('2d loc, 3d scale_tril', (3, 4), 'scale_tril', (1, 4, 4)),
      ('1d loc, 1d scale_diag', (4,), 'scale_diag', (4,)),
      ('1d loc, 2d scale_diag', (4,), 'scale_diag', (3, 4)),
      ('2d loc, 1d scale_diag', (3, 4), 'scale_diag', (4,)),
      ('2d loc, 2d scale_diag', (3, 4), 'scale_diag', (3, 4)),
  )
  def test_shapes(self, loc_shape, scale_type, scale_shape):
    bijector = self._create_random_bijector(loc_shape, scale_type, scale_shape)
    with self.subTest('event_shape'):
      super()._test_event_shape(
          dist_args=(),
          dist_kwargs={'bijector': bijector},
          tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale})
    with self.subTest('batch_shape'):
      super()._test_batch_shape(
          dist_args=(),
          dist_kwargs={'bijector': bijector},
          tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale})

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d loc, 3d scale_tril, no shape', (4,), 'scale_tril', (3, 4, 4), ()),
      ('1d loc, 3d scale_tril, int shape', (4,), 'scale_tril', (3, 4, 4), 2),
      ('1d loc, 3d scale_tril, 2d shape',
       (4,), 'scale_tril', (3, 4, 4), (1, 2)),
      ('2d loc, 3d scale_tril, no shape', (3, 4), 'scale_tril', (1, 4, 4), ()),
      ('2d loc, 3d scale_tril, int shape', (3, 4), 'scale_tril', (1, 4, 4), 2),
      ('2d loc, 3d scale_tril, 2d shape',
       (3, 4), 'scale_tril', (1, 4, 4), (1, 2)),
      ('1d loc, 2d scale_diag, no shape', (4,), 'scale_diag', (3, 4), ()),
      ('1d loc, 2d scale_diag, int shape', (4,), 'scale_diag', (3, 4), 2),
      ('1d loc, 2d scale_diag, 2d shape', (4,), 'scale_diag', (3, 4), (1, 2)),
      ('2d loc, 2d scale_diag, no shape', (3, 4), 'scale_diag', (3, 4), ()),
      ('2d loc, 2d scale_diag, int shape', (3, 4), 'scale_diag', (3, 4), 2),
      ('2d loc, 2d scale_diag, 2d shape', (3, 4), 'scale_diag', (3, 4), (1, 2)),
  )
  def test_sample(self, loc_shape, scale_type, scale_shape, sample_shape):
    bijector = self._create_random_bijector(loc_shape, scale_type, scale_shape)
    with self.subTest('sample_shape'):
      super()._test_sample_shape(
          dist_args=(),
          dist_kwargs={'bijector': bijector},
          tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale},
          sample_shape=sample_shape)
    with self.subTest('sample_and_log_prob'):
      super()._test_sample_and_log_prob(
          dist_args=(),
          dist_kwargs={'bijector': bijector},
          tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale},
          sample_shape=sample_shape,
          assertion_fn=self.assertion_fn)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    bijector = self._create_random_bijector((5,), 'scale_tril', (5, 5))
    dist_params = {'bijector': bijector}
    dist = self.distrax_cls(**dist_params)
    sample_fn = self.variant(
        lambda key: dist.sample(seed=self.key, sample_shape=1))
    samples = sample_fn(self.key)
    chex.assert_type(samples, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d loc, 3d scale_tril, 1d val', (4,), 'scale_tril', (3, 4, 4), (4,)),
      ('1d loc, 3d scale_tril, 2d val', (4,), 'scale_tril', (3, 4, 4), (3, 4)),
      ('1d loc, 3d scale_tril, 3d val',
       (4,), 'scale_tril', (3, 4, 4), (2, 3, 4)),
      ('2d loc, 3d scale_tril, 1d val', (3, 4), 'scale_tril', (1, 4, 4), (4,)),
      ('2d loc, 3d scale_tril, 2d val',
       (3, 4), 'scale_tril', (1, 4, 4), (3, 4)),
      ('2d loc, 3d scale_tril, 3d val',
       (3, 4), 'scale_tril', (1, 4, 4), (2, 3, 4)),
      ('1d loc, 2d scale_diag, 1d val', (4,), 'scale_diag', (3, 4), (4,)),
      ('1d loc, 2d scale_diag, 2d val', (4,), 'scale_diag', (3, 4), (3, 4)),
      ('1d loc, 2d scale_diag, 3d val', (4,), 'scale_diag', (3, 4), (2, 3, 4)),
      ('2d loc, 2d scale_diag, 1d val', (3, 4), 'scale_diag', (3, 4), (4,)),
      ('2d loc, 2d scale_diag, 2d val', (3, 4), 'scale_diag', (3, 4), (3, 4)),
      ('2d loc, 2d scale_diag, 3d val',
       (3, 4), 'scale_diag', (3, 4), (2, 3, 4)),
  )
  def test_log_prob(
      self, loc_shape, scale_type, scale_shape, value_shape):
    bijector = self._create_random_bijector(loc_shape, scale_type, scale_shape)
    value = jax.random.normal(key=self.key_val, shape=value_shape)
    super()._test_attribute(
        attribute_string='log_prob',
        dist_kwargs={'bijector': bijector},
        tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale},
        call_args=(value,),
        assertion_fn=self.assertion_fn)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d loc, 3d scale_tril', (4,), 'scale_tril', (3, 4, 4)),
      ('2d loc, 3d scale_tril', (3, 4), 'scale_tril', (1, 4, 4)),
      ('1d loc, 2d scale_diag', (4,), 'scale_diag', (3, 4)),
      ('2d loc, 2d scale_diag', (3, 4), 'scale_diag', (1, 4)),
  )
  def test_method(self, loc_shape, scale_type, scale_shape):
    bijector = self._create_random_bijector(loc_shape, scale_type, scale_shape)
    for attribute in ['mode', 'variance', 'stddev', 'mean', 'entropy']:
      with self.subTest(attribute):
        super()._test_attribute(
            attribute_string=attribute,
            dist_kwargs={'bijector': bijector},
            tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale},
            assertion_fn=self.assertion_fn)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d loc, 3d scale_tril', (4,), 'scale_tril', (3, 4, 4)),
      ('2d loc, 3d scale_tril', (3, 4), 'scale_tril', (3, 4, 4)),
      ('1d loc, 2d scale_diag', (4,), 'scale_diag', (3, 4)),
      ('2d loc, 2d scale_diag', (3, 4), 'scale_diag', (3, 4)),
  )
  def test_covariance(self, loc_shape, scale_type, scale_shape):
    # Test the covariance separately because it doesn't follow TFP convention.
    bijector = self._create_random_bijector(loc_shape, scale_type, scale_shape)
    for attribute in ['mode', 'variance', 'stddev', 'mean', 'entropy']:
      with self.subTest(attribute):
        super()._test_attribute(
            attribute_string=attribute,
            dist_kwargs={'bijector': bijector},
            tfp_dist_kwargs={'loc': bijector.shift, 'scale': bijector.scale},
            assertion_fn=self.assertion_fn)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d loc, 3d scale_tril', (4,), 'scale_tril', (3, 4, 4)),
      ('2d loc, 3d scale_tril', (3, 4), 'scale_tril', (1, 4, 4)),
      ('1d loc, 2d scale_diag', (4,), 'scale_diag', (3, 4)),
      ('2d loc, 2d scale_diag', (3, 4), 'scale_diag', (3, 4)),
  )
  def test_median(self, loc_shape, scale_type, scale_shape):
    bijector = self._create_random_bijector(loc_shape, scale_type, scale_shape)
    dist_params = {'bijector': bijector}
    dist = self.distrax_cls(**dist_params)
    self.assertion_fn(self.variant(dist.median)(), dist.mean())

  @chex.all_variants
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_two_distributions(self, function_string, mode_string):
    keys = jax.random.split(self.key_bijector, num=2)
    bijector1 = self._create_random_bijector(
        loc_shape=(5,), scale_type='scale_tril', scale_shape=(3, 5, 5),
        key=keys[0])
    bijector2 = self._create_random_bijector(
        loc_shape=(3, 5,), scale_type='scale_tril', scale_shape=(5, 5),
        key=keys[1])
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={'bijector': bijector1},
        tfp_dist1_kwargs={'loc': bijector1.shift, 'scale': bijector1.scale},
        dist2_kwargs={'bijector': bijector2},
        tfp_dist2_kwargs={'loc': bijector2.shift, 'scale': bijector2.scale},
        assertion_fn=self.assertion_fn)

  @parameterized.named_parameters(
      ('scale_tril', ['scale_tril'], False),
      ('scale_diag', ['scale_diag'], True),
      ('scale_identity_multiplier_and_diag',
       ['scale_identity_multiplier', 'scale_diag'], True),
      ('scale_tril_and_diag', ['scale_tril', 'scale_diag'], False),
      ('scale_perturb', ['scale_perturb_factor'], False),
      ('scale_perturb_and_scale_perturb_diag',
       ['scale_perturb_factor', 'scale_perturb_diag'], False),
      ('scale_diag_and_perturb', ['scale_diag', 'scale_perturb_factor'], False),
      ('scale_identity_multiplier_and_tril',
       ['scale_identity_multiplier', 'scale_tril'], False),
  )
  def test_has_diagonal_scale(self, scale_list, is_diag):
    bij_params_dict = dict()
    for elem in scale_list:
      if elem == 'scale_tril':
        bij_params_dict.update({
            'scale_tril': jax.scipy.linalg.tril(
                jnp.ones((4, 4), dtype=jnp.float32)),
        })
      elif elem == 'scale_diag':
        bij_params_dict.update({
            'scale_diag': jnp.ones((4,), dtype=jnp.float32),
        })
      elif elem == 'scale_identity_multiplier':
        bij_params_dict.update({
            'scale_identity_multiplier': 0.5,
        })
      elif elem == 'scale_perturb_factor':
        bij_params_dict.update({
            'scale_perturb_factor': jnp.ones((4, 2), dtype=jnp.float32),
        })
      elif elem == 'scale_perturb_diag':
        bij_params_dict.update({
            'scale_perturb_diag': jnp.ones((2, 2), dtype=jnp.float32),
        })
    bijector = tfb.Affine(
        shift=jnp.zeros((4,), dtype=jnp.float32), **bij_params_dict)
    assert mvn_linear_operator.has_diagonal_scale(bijector) == is_diag
    dist_params = {'bijector': bijector}
    dist = self.distrax_cls(**dist_params)
    assert dist.has_diagonal_scale == is_diag

  def test_jittable(self):
    bijector = tfb.Affine(
        shift=jnp.zeros((4,), dtype=jnp.float32),
        scale_diag=jnp.ones((4,), dtype=jnp.float32))
    super()._test_jittable(
        dist_kwargs={'bijector': bijector},
    )


if __name__ == '__main__':
  absltest.main()
