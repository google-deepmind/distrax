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
"""Wrapper to adapt a TFP distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey
DistributionT = distribution.DistributionT


def distribution_from_tfp(tfp_distribution: tfd.Distribution) -> DistributionT:
  """Create a Distrax distribution from a TFP distribution.

  Given a TFP distribution `tfp_distribution`, this method returns a
  distribution of a class that inherits from the class of `tfp_distribution`.
  The returned distribution behaves almost identically as the TFP distribution,
  except the common methods (`sample`, `variance`, etc.) are overwritten to
  return `jnp.ndarrays`. Moreover, the wrapped distribution also implements
  Distrax methods inherited from `Distribution`, such as `sample_and_log_prob`.

  Args:
    tfp_distribution: A TFP distribution.

  Returns:
    The wrapped distribution.
  """

  class DistributionFromTFP(
      distribution.Distribution, tfp_distribution.__class__):
    """Class to wrap a TFP distribution.

    The wrapped class dynamically inherits from `tfp_distribution`, so that
    computations involving the KL remain valid.
    """

    def __init__(self):
      pass

    def __getattr__(self, name: str):
      return getattr(tfp_distribution, name)

    def sample(self, *a, **k):  # pylint: disable=useless-super-delegation
      """See `Distribution.sample`."""
      return super().sample(*a, **k)

    def _sample_n(self, key: PRNGKey, n: int):
      """See `Distribution._sample_n`."""
      return jnp.asarray(
          tfp_distribution.sample(seed=key, sample_shape=(n,)),
          dtype=tfp_distribution.dtype)

    def log_prob(self, value: Array) -> Array:
      """See `Distribution.log_prob`."""
      return jnp.asarray(tfp_distribution.log_prob(value))

    def prob(self, value: Array) -> Array:
      """See `Distribution.prob`."""
      return jnp.asarray(tfp_distribution.prob(value))

    @property
    def event_shape(self) -> Tuple[int, ...]:
      """See `Distribution.event_shape`."""
      return tuple(tfp_distribution.event_shape)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
      """See `Distribution.batch_shape`."""
      return tuple(tfp_distribution.batch_shape)

    @property
    def name(self) -> str:
      """See `Distribution.name`."""
      return tfp_distribution.name

    @property
    def dtype(self) -> jnp.dtype:
      """See `Distribution.dtype`."""
      return tfp_distribution.dtype

    def kl_divergence(self, other_dist, *args, **kwargs) -> Array:
      """See `Distribution.kl_divergence`."""
      return jnp.asarray(
          tfd.kullback_leibler.kl_divergence(self, other_dist, *args, **kwargs))

    def entropy(self) -> Array:
      """See `Distribution.entropy`."""
      return jnp.asarray(tfp_distribution.entropy())

    def log_cdf(self, value: Array) -> Array:
      """See `Distribution.log_cdf`."""
      return jnp.asarray(tfp_distribution.log_cdf(value))

    def cdf(self, value: Array) -> Array:
      """See `Distribution.cdf`."""
      return jnp.asarray(tfp_distribution.cdf(value))

    def mean(self) -> Array:
      """See `Distribution.mean`."""
      return jnp.asarray(tfp_distribution.mean())

    def median(self) -> Array:
      """See `Distribution.median`."""
      return jnp.asarray(tfp_distribution.median())

    def variance(self) -> Array:
      """See `Distribution.variance`."""
      return jnp.asarray(tfp_distribution.variance())

    def stddev(self) -> Array:
      """See `Distribution.stddev`."""
      return jnp.asarray(tfp_distribution.stddev())

    def mode(self) -> Array:
      """See `Distribution.mode`."""
      return jnp.asarray(tfp_distribution.mode())

    def __getitem__(self, index) -> DistributionT:
      """See `Distribution.__getitem__`."""
      return distribution_from_tfp(tfp_distribution[index])

  return DistributionFromTFP()
