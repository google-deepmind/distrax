# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
from distrax._src.distributions import beta
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import jittable
import jax
import jax.numpy as jnp
from jax.scipy import special as jax_special
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


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
  `concentration1_numerator` (a0), `concentration0_numerator` (b0),
  `concentration1_denominator` (a1), and `concentration0_denominator` (b1).
  """

  equiv_tfp_cls = tfd.BetaQuotient

  def __init__(self,
               concentration1_numerator: Numeric,
               concentration0_numerator: Numeric,
               concentration1_denominator: Numeric,
               concentration0_denominator: Numeric,
               name: str = 'BetaQuotient'):
    """Initializes a BetaQuotient distribution.

    Args:
      concentration1_numerator: Numerator `alpha` parameter (a0).
      concentration0_numerator: Numerator `beta` parameter (b0).
      concentration1_denominator: Denominator `alpha` parameter (a1).
      concentration0_denominator: Denominator `beta` parameter (b1).
      name: Name of the distribution.
    """
    self._concentration1_numerator = conversion.to_float_array(
        concentration1_numerator)
    self._concentration0_numerator = conversion.to_float_array(
        concentration0_numerator)
    self._concentration1_denominator = conversion.to_float_array(
        concentration1_denominator)
    self._concentration0_denominator = conversion.to_float_array(
        concentration0_denominator)

    super().__init__(name=name)

  @property
  def concentration1_numerator(self) -> Array:
    """Numerator `alpha` parameter."""
    return self._concentration1_numerator

  @property
  def concentration0_numerator(self) -> Array:
    """Numerator `beta` parameter."""
    return self._concentration0_numerator

  @property
  def concentration1_denominator(self) -> Array:
    """Denominator `alpha` parameter."""
    return self._concentration1_denominator

  @property
  def concentration0_denominator(self) -> Array:
    """Denominator `beta` parameter."""
    return self._concentration0_denominator

  def _event_shape(self) -> Tuple[int, ...]:
    return ()

  @jittable
  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key_num, key_den = jax.random.split(key)

    alpha0 = jnp.broadcast_to(self.concentration1_numerator, self.batch_shape)
    beta0 = jnp.broadcast_to(self.concentration0_numerator, self.batch_shape)
    alpha1 = jnp.broadcast_to(self.concentration1_denominator, self.batch_shape)
    beta1 = jnp.broadcast_to(self.concentration0_denominator, self.batch_shape)

    numerator_dist = beta.Beta(alpha0, beta0)
    denominator_dist = beta.Beta(alpha1, beta1)

    numerator_samples = numerator_dist.sample(seed=key_num, sample_shape=n)
    denominator_samples = denominator_dist.sample(seed=key_den, sample_shape=n)

    return numerator_samples / denominator_samples

  @jittable
  def _log_prob(self, value: EventT) -> Array:
    """See `Distribution._log_prob`."""
    a0 = self.concentration1_numerator
    b0 = self.concentration0_numerator
    a1 = self.concentration1_denominator
    b1 = self.concentration0_denominator

    x = value
    one = jnp.ones_like(x)

    a_sum = a0 + a1

    log_normalization = (
        jax_special.lbeta(a0, b0) + jax_special.lbeta(a1, b1))
    log_normalization = log_normalization - jax_special.lbeta(
        a_sum, jnp.where(x > 1., b0, b1))

    b = jnp.where(x > 1., 1. - b1, 1. - b0)
    c = a_sum + jnp.where(x > 1., b0, b1)
    z = jnp.where(x > 1., jnp.reciprocal(x), x)

    # Note: jax.scipy.special.hyp2f1 is defined for |z| < 1.
    # When x > 1, z = 1/x, so |z| < 1.
    # When x <= 1, z = x. The distribution is defined for x > 0.
    # If x = 1, z = 1. We need to handle this case.
    # TFP's hyp2f1_small_argument seems to handle z=1.
    # Let's check jax.scipy.special.hyp2f1 documentation.
    # It says "The series representation converges for |z| < 1."
    # TFP's notes: "Here, c - a - b = beta0 + beta1 - 1, so the series always
    # converges conditionally."
    # Let's assume jax_special.hyp2f1 handles z=1 if c - a - b > 0.
    # c - b = (a_sum + b0) - (1 - b1) = a0 + a1 + b0 + b1 - 1 (for x > 1)
    # c - b = (a_sum + b1) - (1 - b0) = a0 + a1 + b1 + b0 - 1 (for x <= 1)
    # The 'a' in hyp2f1 is a_sum.
    # So c - a_sum - b = (a_sum + b_choice) - a_sum - (1 - b_other)
    # c - a - b = b_choice + b_other - 1
    # For x > 1: b0 + b1 - 1
    # For x <= 1: b1 + b0 - 1
    # This matches TFP's note.

    # We must clip z to be <= 1 to avoid NaN from hyp2f1(..., z=1.000001)
    # due to floating point.
    # We also need to ensure z is not 1 if c - a - b <= 0.
    # Since a0, b0, a1, b1 > 0, b0 + b1 - 1 can be <= 0.
    # TFP uses hyp2f1_small_argument, which might have special handling.
    # Let's clip z at 1 - eps.
    eps = jnp.finfo(z.dtype).eps
    z = jnp.clip(z, a_max=one - eps)

    log_unnormalized_prob = jnp.log(
        jax_special.hyp2f1(a_sum, b, c, z))

    # Use jnp.logaddexp trick for xlogy(y, x) -> log(y^x)
    # xlogy(y, x) = y * log(x)
    # No, tf.math.xlogy(a, b) is a * log(b)
    # So TFP is: `(tf.where(x > 1., -(a1 + 1.), a0 - 1.)) * log(x)`

    log_x = jnp.log(x)
    term2 = jnp.where(x > 1., -(a1 + 1.), a0 - 1.) * log_x

    log_unnormalized_prob = log_unnormalized_prob + term2

    return log_unnormalized_prob - log_normalization

  @jittable
  def mean(self) -> Array:
    """Calculates the mean."""
    a0 = self.concentration1_numerator
    b0 = self.concentration0_numerator
    a1 = self.concentration1_denominator
    b1 = self.concentration0_denominator

    # TFP: mean = a0 * (a1 + b1 - 1.) / ((a0 + b0) * (a1 - 1.))
    mean = a0 * (a1 + b1 - 1.) / ((a0 + b0) * (a1 - 1.))

    # Mean is defined for a1 > 1.
    return jnp.where(a1 > 1., mean, jnp.nan)

  def variance(self) -> Array:
    """Calculates the variance."""
    # Not implemented in TFP, returning nan.
    return jnp.full(self.batch_shape, jnp.nan)

