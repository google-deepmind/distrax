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
"""Tests for `negative_binomial.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import negative_binomial
from distrax._src.utils import equivalence
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special as sp_special


class NegativeBinomialTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(negative_binomial.NegativeBinomial)
    self.p = np.asarray([0.2, 0.4, 0.6, 0.8])
    self.logits = sp_special.logit(self.p)

  def test_raises_on_invalid_inputs(self, dist_params):
    with self.assertRaises(ValueError):
      self.distrax_cls(**dist_params)


  @parameterized.named_parameters(
      ('0d params', (), (), ()),
      ('1d params', (2,), (2,), (2,)),
      ('2d params, no broadcast', (3, 2), (3, 2), (3, 2)),
      ('2d params, broadcasted total_count', (2,), (3, 2), (3, 2)),
      ('2d params, broadcasted probs', (3, 2), (2,), (3, 2)),
  )
  def test_properties_probs(self, total_count_shape, probs_shape, batch_shape):
    rng = np.random.default_rng(42)
    total_count = rng.integers(0, 10e3, size=total_count_shape)
    probs = rng.uniform(size=probs_shape)
    dist = self.distrax_cls(total_count, probs=probs) 
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=2e-2)(
        dist.total_count, np.broadcast_to(total_count, batch_shape))
    self.assertion_fn(rtol=2e-2)(dist.probs, np.broadcast_to(probs, batch_shape))

  @parameterized.named_parameters(
      ('0d params', (), (), ()),
      ('1d params', (2,), (2,), (2,)),
      ('2d params, no broadcast', (3, 2), (3, 2), (3, 2)),
      ('2d params, broadcasted total_count', (2,), (3, 2), (3, 2)),
      ('2d params, broadcasted logits', (3, 2), (2,), (3, 2)),
  )
  def test_properties_logits(self, total_count_shape, logits_shape, batch_shape):
    rng = np.random.default_rng(42)
    total_count = rng.integers(0, 10e3, size=total_count_shape)
    logits = rng.uniform(size=logits_shape)
    dist = self.distrax_cls(total_count, logits=logits) 
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=2e-2)(
        dist.total_count, np.broadcast_to(total_count, batch_shape))
    self.assertion_fn(rtol=2e-2)(dist.logits, np.broadcast_to(logits, batch_shape))
  
  @parameterized.named_parameters(
      ('0d params', (), (), ()),
      ('1d params', (2,), (2,), (2,)),
      ('2d params, no broadcast', (3, 2), (3, 2), (3, 2)),
      ('2d params, broadcasted mean', (2,), (3, 2), (3, 2)),
      ('2d params, broadcasted dispersion', (3, 2), (2,), (3, 2)),
  )
  def test_properties_mean_dispersion(self, mean_shape, dispersion_shape, batch_shape):
    rng = np.random.default_rng(42)
    mean = 1000. * rng.uniform(size=mean_shape)
    dispersion = rng.uniform(size=dispersion_shape)
    dist = self.distrax_cls.from_mean_dispersion(
      mean=mean, dispersion=dispersion) 
    self.assertEqual(dist.event_shape, ())
    self.assertEqual(dist.batch_shape, batch_shape)
    self.assertion_fn(rtol=2e-2)(
        dist.mean(), np.broadcast_to(mean, batch_shape))
    self.assertion_fn(rtol=2e-2)(dist.dispersion, np.broadcast_to(dispersion, batch_shape))

if __name__ == '__main__':
  absltest.main()