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
"""Poisson distribution."""

from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class Poisson(distribution.Distribution):
  """Poisson distribution with a rate parameter."""

  equiv_tfp_cls = tfd.Poisson

  def __init__(self, rate: Numeric):
    """Initializes a Poisson distribution.

    Args:
      rate: Rate of the distribution.
    """
    super().__init__()
    self._rate = conversion.as_float_array(rate)
    self._batch_shape = self._rate.shape

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def rate(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._rate, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    out_shape = (n,) + self.batch_shape
    return jax.random.poisson(key, self.rate, out_shape)

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    return (
        (jnp.log(self.rate) * value)
        - jax.scipy.special.gammaln(value + 1)
        - self.rate
    )

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    x = jnp.floor(value) + 1
    return jax.scipy.special.gammaincc(x, self.rate)

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.rate

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return jnp.sqrt(self.rate)

  def variance(self) -> Array:
    """Calculates the variance."""
    return self.rate

  def mode(self) -> Array:
    """Calculates the mode."""
    return jnp.ceil(self.rate) - 1

  def __getitem__(self, index) -> 'Poisson':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Poisson(rate=self.rate[index])


def _kl_divergence_poisson_poisson(
    dist1: Union[Poisson, tfd.Poisson],
    dist2: Union[Poisson, tfd.Poisson],
    *unused_args,
    **unused_kwargs,
) -> Array:
  """Batched KL divergence KL(dist1 || dist2) between two poisson distributions.

  Args:
    dist1: A poisson distribution.
    dist2: A poisson distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  distance = dist1.rate - dist2.rate
  diff_log_scale = jnp.log(dist1.rate) - jnp.log(dist2.rate)
  return math.multiply_no_nan(dist1.rate, diff_log_scale) - distance


# Register the KL functions with TFP.
tfd.RegisterKL(Poisson, Poisson)(_kl_divergence_poisson_poisson)
tfd.RegisterKL(Poisson, Poisson.equiv_tfp_cls)(_kl_divergence_poisson_poisson)
tfd.RegisterKL(Poisson.equiv_tfp_cls, Poisson)(_kl_divergence_poisson_poisson)
