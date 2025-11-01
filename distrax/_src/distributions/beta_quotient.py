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
"""BetaQuotient distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import beta as beta_lib
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import jax.scipy.special
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class BetaQuotient(distribution.Distribution):
  """BetaQuotient distribution.

  The Beta Quotient distribution is defined over the positive reals, as
  the ratio of two Independent Beta distributed random variables.

  In other words:

  ```none
  X ~ Beta(a0, b0)
  Y ~ Beta(a1, b1)
  X / Y ~ BetaQuotient(a0, b0, a1, b1)
  ```

  The distribution is defined over the positive reals, by four parameters
  `concentration0_numerator`, `concentration1_numerator`,
  `concentration0_denominator` and `concentration1_denominator`
  (aka `beta` and `alpha` of the numerator and denominator Beta distribution
  respectively).
  """

  equiv_tfp_cls = tfd.BetaQuotient

  def __init__(
      self,
      concentration1_numerator: Numeric,
      concentration0_numerator: Numeric,
      concentration1_denominator: Numeric,
      concentration0_denominator: Numeric,
  ):
    """Initializes a BetaQuotient distribution.

    Args:
      concentration1_numerator: Numerator shape parameter `alpha` (aka `a0`).
        Must be positive.
      concentration0_numerator: Numerator shape parameter `beta` (aka `b0`).
        Must be positive.
      concentration1_denominator: Denominator shape parameter `alpha` (aka
        `a1`). Must be positive.
      concentration0_denominator: Denominator shape parameter `beta` (aka `b1`).
        Must be positive.
    """
    super().__init__()
    # By default, distrax converts to float32
    self._concentration1_numerator = conversion.as_float_array(
        concentration1_numerator
    )
    self._concentration0_numerator = conversion.as_float_array(
        concentration0_numerator
    )
    self._concentration1_denominator = conversion.as_float_array(
        concentration1_denominator
    )
    self._concentration0_denominator = conversion.as_float_array(
        concentration0_denominator
    )

    self._batch_shape = jax.lax.broadcast_shapes(
        self._concentration1_numerator.shape,
        self._concentration0_numerator.shape,
        self._concentration1_denominator.shape,
        self._concentration0_denominator.shape,
    )

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def concentration1_numerator(self) -> Array:
    """Concentration parameter associated with a `1` outcome."""
    return jnp.broadcast_to(self._concentration1_numerator, self.batch_shape)

  @property
  def concentration0_numerator(self) -> Array:
    """Concentration parameter associated with a `0` outcome."""
    return jnp.broadcast_to(self._concentration0_numerator, self.batch_shape)

  @property
  def concentration1_denominator(self) -> Array:
    """Concentration parameter associated with a `1` outcome."""
    return jnp.broadcast_to(self._concentration1_denominator, self.batch_shape)

  @property
  def concentration0_denominator(self) -> Array:
    """Concentration parameter associated with a `0` outcome."""
    return jnp.broadcast_to(self._concentration0_denominator, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key1, key2 = jax.random.split(key)
    numerator = beta_lib.Beta(
        self.concentration1_numerator, self.concentration0_numerator
    )
    denominator = beta_lib.Beta(
        self.concentration1_denominator, self.concentration0_denominator
    )
    return numerator.sample(seed=key1, sample_shape=(n,)) / denominator.sample(
        seed=key2, sample_shape=(n,)
    )

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    a0 = self.concentration1_numerator
    b0 = self.concentration0_numerator
    a1 = self.concentration1_denominator
    b1 = self.concentration0_denominator

    alpha_sum = a0 + a1
    x = value

    log_normalization = math.log_beta(a0, b0) + math.log_beta(a1, b1)
    log_normalization = log_normalization - math.log_beta(
        alpha_sum, jnp.where(x > 1.0, b0, b1)
    )

    b = jnp.where(x > 1.0, 1.0 - b1, 1 - b0)
    c = alpha_sum + jnp.where(x > 1.0, b0, b1)
    z = jnp.where(x > 1.0, jnp.reciprocal(x), x)

    # Here, c - a - b = b0 + b1 - 1, so the series always converges
    # conditionally.
    log_unnormalized_prob = jnp.log(
        jax.scipy.special.hyp2f1(alpha_sum, b, c, z)
    )
    log_unnormalized_prob = log_unnormalized_prob + jax.scipy.special.xlogy(
        jnp.where(x > 1.0, -(a1 + 1.0), a0 - 1.0), x
    )

    return log_unnormalized_prob - log_normalization

  def mean(self) -> Array:
    """Calculates the mean."""
    a0 = self.concentration1_numerator
    b0 = self.concentration0_numerator
    a1 = self.concentration1_denominator
    b1 = self.concentration0_denominator
    mean = a0 * (a1 + b1 - 1.0) / ((a0 + b0) * (a1 - 1.0))
    # Ensure promotion to float32 (default)
    mean = mean.astype(jnp.result_type(a0, b0, a1, b1))
    return jnp.where(a1 > 1.0, mean, jnp.nan)

  def __getitem__(self, index) -> 'BetaQuotient':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return BetaQuotient(
        concentration1_numerator=self.concentration1_numerator[index],
        concentration0_numerator=self.concentration0_numerator[index],
        concentration1_denominator=self.concentration1_denominator[index],
        concentration0_denominator=self.concentration0_denominator[index],
    )
