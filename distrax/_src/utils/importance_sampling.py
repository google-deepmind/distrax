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
"""Importance sampling."""

import chex
from distrax._src.distributions import distribution
import jax.numpy as jnp


Array = chex.Array
DistributionLike = distribution.DistributionLike


def importance_sampling_ratios(
    target_dist: DistributionLike,
    sampling_dist: DistributionLike,
    event: Array
) -> Array:
  """Compute importance sampling ratios given target and sampling distributions.

  Args:
    target_dist: Target probability distribution.
    sampling_dist: Sampling probability distribution.
    event: Samples.

  Returns:
    Importance sampling ratios.
  """
  log_pi_a = target_dist.log_prob(event)
  log_mu_a = sampling_dist.log_prob(event)
  rho = jnp.exp(log_pi_a - log_mu_a)
  return rho
