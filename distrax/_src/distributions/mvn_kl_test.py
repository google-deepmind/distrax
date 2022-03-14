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
"""Tests for `kl_divergence` across different types of MultivariateNormal."""

from absl.testing import absltest
from absl.testing import parameterized

import chex

from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from distrax._src.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from distrax._src.distributions.mvn_tri import MultivariateNormalTri

import numpy as np


def _get_dist_params(dist, batch_shape, dim, rng):
  """Generates random parameters depending on the distribution type."""
  if dist is MultivariateNormalDiag:
    distrax_dist_params = {
        'scale_diag': rng.normal(size=batch_shape + (dim,)),
    }
    tfp_dist_params = distrax_dist_params
  elif dist is MultivariateNormalDiagPlusLowRank:
    scale_diag = rng.normal(size=batch_shape + (dim,))
    scale_u_matrix = 0.2 * rng.normal(size=batch_shape + (dim, 2))
    scale_perturb_diag = rng.normal(size=batch_shape + (2,))
    scale_v_matrix = scale_u_matrix * np.expand_dims(
        scale_perturb_diag, axis=-2)
    distrax_dist_params = {
        'scale_diag': scale_diag,
        'scale_u_matrix': scale_u_matrix,
        'scale_v_matrix': scale_v_matrix,
    }
    tfp_dist_params = {
        'scale_diag': scale_diag,
        'scale_perturb_factor': scale_u_matrix,
        'scale_perturb_diag': scale_perturb_diag,
    }
  elif dist is MultivariateNormalTri:
    scale_tril = rng.normal(size=batch_shape + (dim, dim))
    distrax_dist_params = {
        'scale_tri': scale_tril,
        'is_lower': True,
    }
    tfp_dist_params = {
        'scale_tril': scale_tril,
    }
  elif dist is MultivariateNormalFullCovariance:
    matrix = rng.normal(size=batch_shape + (dim, dim))
    matrix_t = np.vectorize(np.transpose, signature='(k,k)->(k,k)')(matrix)
    distrax_dist_params = {
        'covariance_matrix': np.matmul(matrix, matrix_t),
    }
    tfp_dist_params = distrax_dist_params
  loc = rng.normal(size=batch_shape + (dim,))
  distrax_dist_params.update({'loc': loc})
  tfp_dist_params.update({'loc': loc})
  return distrax_dist_params, tfp_dist_params


class MultivariateNormalKLTest(parameterized.TestCase):

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('Diag vs DiagPlusLowRank',
       MultivariateNormalDiag, MultivariateNormalDiagPlusLowRank),
      ('Diag vs FullCovariance',
       MultivariateNormalDiag, MultivariateNormalFullCovariance),
      ('Diag vs Tri',
       MultivariateNormalDiag, MultivariateNormalTri),
      ('DiagPlusLowRank vs FullCovariance',
       MultivariateNormalDiagPlusLowRank, MultivariateNormalFullCovariance),
      ('DiagPlusLowRank vs Tri',
       MultivariateNormalDiagPlusLowRank, MultivariateNormalTri),
      ('Tri vs FullCovariance',
       MultivariateNormalTri, MultivariateNormalFullCovariance),
  )
  def test_two_distributions(self, dist1_type, dist2_type):
    rng = np.random.default_rng(42)

    distrax_dist1_params, tfp_dist1_params = _get_dist_params(
        dist1_type, batch_shape=(8, 1), dim=5, rng=rng)
    distrax_dist2_params, tfp_dist2_params = _get_dist_params(
        dist2_type, batch_shape=(6,), dim=5, rng=rng)

    dist1_distrax = dist1_type(**distrax_dist1_params)
    dist1_tfp = dist1_type.equiv_tfp_cls(**tfp_dist1_params)
    dist2_distrax = dist2_type(**distrax_dist2_params)
    dist2_tfp = dist2_type.equiv_tfp_cls(**tfp_dist2_params)

    for method in ['kl_divergence', 'cross_entropy']:
      expected_result1 = getattr(dist1_tfp, method)(dist2_tfp)
      expected_result2 = getattr(dist2_tfp, method)(dist1_tfp)
      for mode in ['distrax_to_distrax', 'distrax_to_tfp', 'tfp_to_distrax']:
        with self.subTest(method=method, mode=mode):
          if mode == 'distrax_to_distrax':
            result1 = self.variant(getattr(dist1_distrax, method))(
                dist2_distrax)
            result2 = self.variant(getattr(dist2_distrax, method))(
                dist1_distrax)
          elif mode == 'distrax_to_tfp':
            result1 = self.variant(getattr(dist1_distrax, method))(dist2_tfp)
            result2 = self.variant(getattr(dist2_distrax, method))(dist1_tfp)
          elif mode == 'tfp_to_distrax':
            result1 = self.variant(getattr(dist1_tfp, method))(dist2_distrax)
            result2 = self.variant(getattr(dist2_tfp, method))(dist1_distrax)
          np.testing.assert_allclose(result1, expected_result1, rtol=0.02)
          np.testing.assert_allclose(result2, expected_result2, rtol=0.02)


if __name__ == '__main__':
  absltest.main()
