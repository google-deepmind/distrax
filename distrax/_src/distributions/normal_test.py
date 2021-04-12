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
"""Tests for `normal.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import normal
from distrax._src.utils import equivalence
import numpy as np


RTOL = 1e-2


class NormalTest(equivalence.EquivalenceTest, parameterized.TestCase):

  def setUp(self):
    # pylint: disable=too-many-function-args
    super().setUp(normal.Normal)
    self.assertion_fn = lambda x, y: np.testing.assert_allclose(x, y, rtol=RTOL)

  @parameterized.named_parameters(
      ('1d std normal', (0, 1)),
      ('2d std normal', (np.zeros(2), np.ones(2))),
      ('rank 2 std normal', (np.zeros((3, 2)), np.ones((3, 2)))),
      ('broadcasted loc', (0, np.ones(3))),
      ('broadcasted scale', (np.ones(3), 1)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std normal, no shape', (0, 1), ()),
      ('1d std normal, int shape', (0, 1), 1),
      ('1d std normal, 1-tuple shape', (0, 1), (1,)),
      ('1d std normal, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std normal, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std normal, int shape', ([0, 0], [1, 1]), 1),
      ('2d std normal, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std normal, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
      ('rank 2 std normal, 2-tuple shape', (np.zeros((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted loc', (0, np.ones(3)), (2, 2)),
      ('broadcasted scale', (np.ones(3), 1), ()),
  )
  def test_sample_shape(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d std normal, no shape', (0, 1), ()),
      ('1d std normal, int shape', (0, 1), 1),
      ('1d std normal, 1-tuple shape', (0, 1), (1,)),
      ('1d std normal, 2-tuple shape', (0, 1), (2, 2)),
      ('2d std normal, no shape', (np.zeros(2), np.ones(2)), ()),
      ('2d std normal, int shape', ([0, 0], [1, 1]), 1),
      ('2d std normal, 1-tuple shape', (np.zeros(2), np.ones(2)), (1,)),
      ('2d std normal, 2-tuple shape', ([0, 0], [1, 1]), (2, 2)),
      ('rank 2 std normal, 2-tuple shape', (np.zeros((3, 2)), np.ones(
          (3, 2))), (2, 2)),
      ('broadcasted loc', (0, np.ones(3)), (2, 2)),
      ('broadcasted scale', (np.ones(3), 1), ()),
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_sample_and_log_prob(
        dist_args=distr_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d dist, 1d value', (0, 1), 1),
      ('1d dist, 2d value', (0., 1.), np.array([1, 2])),
      ('2d dist, 1d value', (np.zeros(2), np.ones(2)), 1),
      ('2d broadcasted dist, 1d value', (np.zeros(2), 1), 1),
      ('2d dist, 2d value', (np.zeros(2), np.ones(2)), np.array([1, 2])),
      ('1d dist, 1d value, edge case', (0, 1), 200),
  )
  def test_method_with_input(self, distr_params, value):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    value = np.asarray(value, dtype=np.float32)
    for method in ['log_prob', 'prob', 'cdf', 'log_cdf']:
      with self.subTest(method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=distr_params,
            dist_kwargs={},
            call_args=(value,),
            assertion_fn=self.assertion_fn)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', (0., 1.), 'entropy'),
      ('mean', (0, 1), 'mean'),
      ('mean from 1d params', ([-1, 1], [1, 2]), 'mean'),
      ('variance', (0, 1), 'variance'),
      ('variance from np params', (np.ones(2), np.ones(2)), 'variance'),
      ('stddev', (0, 1), 'stddev'),
      ('stddev from rank 2 params', (np.ones((2, 3)), np.ones(
          (2, 3))), 'stddev'),
      ('mode', (0, 1), 'mode'),
  )
  def test_method(self, distr_params, function_string):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    super()._test_attribute(
        function_string,
        distr_params,
        assertion_fn=self.assertion_fn)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('no broadcast', ([0., 1., -0.5], [0.5, 1., 1.5])),
      ('broadcasted loc', (0.5, [0.5, 1., 1.5])),
      ('broadcasted scale', ([0., 1., -0.5], 0.8)),
  )
  def test_median(self, distr_params):
    distr_params = (np.asarray(distr_params[0], dtype=np.float32),
                    np.asarray(distr_params[1], dtype=np.float32))
    dist = self.distrax_cls(*distr_params)
    self.assertion_fn(self.variant(dist.median)(), dist.mean())

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'kl_divergence', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'kl_divergence', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'kl_divergence', 'tfp_to_distrax'),
      ('cross-ent distrax_to_distrax', 'cross_entropy', 'distrax_to_distrax'),
      ('cross-ent distrax_to_tfp', 'cross_entropy', 'distrax_to_tfp'),
      ('cross-ent tfp_to_distrax', 'cross_entropy', 'tfp_to_distrax'))
  def test_with_two_distributions(self, function_string, mode_string):
    super()._test_with_two_distributions(
        attribute_string=function_string,
        mode_string=mode_string,
        dist1_kwargs={
            'loc': np.random.randn(4, 1, 2),
            'scale': np.asarray([[0.8, 0.2], [0.1, 1.2], [1.4, 3.1]]),
        },
        dist2_kwargs={
            'loc': np.random.randn(3, 2),
            'scale': 0.1 + np.random.rand(4, 1, 2),
        },
        assertion_fn=self.assertion_fn)

  def test_jittable(self):
    super()._test_jittable((np.zeros((3,)), np.ones((3,))))


if __name__ == '__main__':
  absltest.main()
