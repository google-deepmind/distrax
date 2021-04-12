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
"""Tests for `importance_sampling.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import categorical
from distrax._src.utils import importance_sampling
import jax
import jax.numpy as jnp
import numpy as np


class ImportanceSamplingTest(parameterized.TestCase):

  @chex.all_variants(with_pmap=False)
  def test_importance_sampling_ratios_on_policy(self):
    key = jax.random.PRNGKey(42)
    probs = jnp.array([0.4, 0.2, 0.1, 0.3])
    dist = categorical.Categorical(probs=probs)
    event = dist.sample(seed=key, sample_shape=())

    ratios_fn = self.variant(
        importance_sampling.importance_sampling_ratios)
    rhos = ratios_fn(target_dist=dist, sampling_dist=dist, event=event)

    expected_rhos = jnp.ones_like(rhos)
    np.testing.assert_array_almost_equal(rhos, expected_rhos)

  @chex.all_variants(with_pmap=False)
  def test_importance_sampling_ratios_off_policy(self):
    """Tests for a full batch."""
    pi_logits = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    pi = categorical.Categorical(logits=pi_logits)
    mu_logits = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    mu = categorical.Categorical(logits=mu_logits)
    events = np.array([1, 0], dtype=np.int32)

    ratios_fn = self.variant(
        importance_sampling.importance_sampling_ratios)
    rhos = ratios_fn(pi, mu, events)

    expected_rhos = np.array(
        [pi.probs[0][1] / mu.probs[0][1], pi.probs[1][0] / mu.probs[1][0]],
        dtype=np.float32)
    np.testing.assert_allclose(expected_rhos, rhos, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
