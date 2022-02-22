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
"""Monte-Carlo estimation of the KL divergence."""

from typing import Optional

import chex
from distrax._src.distributions.distribution import DistributionLike
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
PRNGKey = chex.PRNGKey


def estimate_kl_best_effort(
    distribution_a: DistributionLike,
    distribution_b: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int,
    proposal_distribution: Optional[DistributionLike] = None):
  """Estimates KL(distribution_a, distribution_b) exactly or with DiCE.

  If the kl_divergence(distribution_a, distribution_b) is not supported,
  the DiCE estimator is used instead.

  Args:
    distribution_a: The first distribution.
    distribution_b: The second distribution.
    rng_key: The PRNGKey random key.
    num_samples: The number of samples, if using the DiCE estimator.
    proposal_distribution: A proposal distribution for the samples, if using
      the DiCE estimator. If None, use `distribution_a` as proposal.

  Returns:
    The estimated KL divergence.
  """
  distribution_a = conversion.as_distribution(distribution_a)
  distribution_b = conversion.as_distribution(distribution_b)
  # If possible, compute the exact KL.
  try:
    return tfd.kl_divergence(distribution_a, distribution_b)
  except NotImplementedError:
    pass
  return mc_estimate_kl(distribution_a, distribution_b, rng_key,
                        num_samples=num_samples,
                        proposal_distribution=proposal_distribution)


def mc_estimate_kl(
    distribution_a: DistributionLike,
    distribution_b: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int,
    proposal_distribution: Optional[DistributionLike] = None):
  """Estimates KL(distribution_a, distribution_b) with the DiCE estimator.

  To get correct gradients with respect the `distribution_a`, we use the DiCE
  estimator, i.e., we stop the gradient with respect to the samples and with
  respect to the denominator in the importance weights. We then do not need
  reparametrized distributions.

  Args:
    distribution_a: The first distribution.
    distribution_b: The second distribution.
    rng_key: The PRNGKey random key.
    num_samples: The number of samples, if using the DiCE estimator.
    proposal_distribution: A proposal distribution for the samples, if using the
      DiCE estimator. If None, use `distribution_a` as proposal.

  Returns:
    The estimated KL divergence.
  """
  if proposal_distribution is None:
    proposal_distribution = distribution_a
  proposal_distribution = conversion.as_distribution(proposal_distribution)
  distribution_a = conversion.as_distribution(distribution_a)
  distribution_b = conversion.as_distribution(distribution_b)

  samples, logp_proposal = proposal_distribution.sample_and_log_prob(
      seed=rng_key, sample_shape=[num_samples])
  samples = jax.lax.stop_gradient(samples)
  logp_proposal = jax.lax.stop_gradient(logp_proposal)
  logp_a = distribution_a.log_prob(samples)
  logp_b = distribution_b.log_prob(samples)
  importance_weight = jnp.exp(logp_a - logp_proposal)
  log_ratio = logp_b - logp_a
  kl_estimator = -importance_weight * log_ratio
  return jnp.mean(kl_estimator, axis=0)


def mc_estimate_kl_with_reparameterized(
    distribution_a: DistributionLike,
    distribution_b: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int):
  """Estimates KL(distribution_a, distribution_b)."""
  if isinstance(distribution_a, tfd.Distribution):
    if distribution_a.reparameterization_type != tfd.FULLY_REPARAMETERIZED:
      raise ValueError(
          f'Distribution `{distribution_a.name}` cannot be reparameterized.')
  distribution_a = conversion.as_distribution(distribution_a)
  distribution_b = conversion.as_distribution(distribution_b)

  samples, logp_a = distribution_a.sample_and_log_prob(
      seed=rng_key, sample_shape=[num_samples])
  logp_b = distribution_b.log_prob(samples)
  log_ratio = logp_b - logp_a
  kl_estimator = -log_ratio
  return jnp.mean(kl_estimator, axis=0)


def mc_estimate_mode(
    distribution: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int):
  """Returns a Monte Carlo estimate of the mode of a distribution."""
  distribution = conversion.as_distribution(distribution)
  # Obtain samples from the distribution and their log probability.
  samples, log_probs = distribution.sample_and_log_prob(
      seed=rng_key, sample_shape=[num_samples])
  # Do argmax over the sample_shape.
  index = jnp.expand_dims(jnp.argmax(log_probs, axis=0), axis=0)
  # Broadcast index to include event_shape of the sample.
  index = index.reshape(index.shape + (1,) * (samples.ndim - index.ndim))
  mode = jnp.squeeze(jnp.take_along_axis(samples, index, axis=0), axis=0)
  return mode
