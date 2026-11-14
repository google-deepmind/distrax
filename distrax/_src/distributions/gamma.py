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
"""Gamma distribution."""

from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class Gamma(distribution.Distribution):
  """Gamma distribution with parameters `concentration` and `rate`."""

  equiv_tfp_cls = tfd.Gamma

  def __init__(self, concentration: Numeric, rate: Numeric):
    """Initializes a Gamma distribution.

    Args:
      concentration: Concentration parameter of the distribution.
      rate: Inverse scale params of the distribution.
    """
    super().__init__()
    self._concentration = conversion.as_float_array(concentration)
    self._rate = conversion.as_float_array(rate)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._concentration.shape, self._rate.shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def concentration(self) -> Array:
    """Concentration of the distribution."""
    return jnp.broadcast_to(self._concentration, self.batch_shape)

  @property
  def rate(self) -> Array:
    """Inverse scale of the distribution."""
    return jnp.broadcast_to(self._rate, self.batch_shape)

  def _sample_from_std_gamma(self, key: PRNGKey, n: int) -> Array:
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._concentration, self._rate)
    return jax.random.gamma(
        key, a=self._concentration, shape=out_shape, dtype=dtype
    )

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    rnd = self._sample_from_std_gamma(key, n)
    return rnd / self._rate

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    return (
        self._concentration * jnp.log(self._rate)
        + (self._concentration - 1) * jnp.log(value)
        - self._rate * value
        - jax.lax.lgamma(self._concentration)
    )

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    dtype = jnp.result_type(self._concentration, self._rate)
    concentration = jnp.asarray(self._concentration, dtype=dtype)
    log_rate = jnp.log(jnp.asarray(self._rate, dtype=dtype))
    # Combine the Stirling expansions before evaluating the cancelling terms.
    # For a >= 2, the omitted terms are bounded by 1 / (126 * a**6).
    cutoff = max(2., float(
        (1. / (126. * float(jnp.finfo(dtype).eps))) ** (1. / 6.)))
    use_expansion = concentration >= cutoff
    # Mask both branches before arithmetic so inactive derivatives stay finite.
    small = jnp.where(use_expansion, 1., concentration)
    large = jnp.where(use_expansion, concentration, 1.)
    regular = (small - log_rate + jax.lax.lgamma(small)
               + (1. - small) * jax.lax.digamma(small))
    inverse = 1. / large
    correction = inverse * (-1. / 3. + inverse * (
        -1. / 12. + inverse * (-1. / 90. + inverse * (
            1. / 120. + inverse / 210.))))
    expansion = (0.5 * jnp.log(large) - log_rate
                 + 0.5 * (1. + jnp.log(2. * jnp.pi)) + correction)
    return jnp.where(use_expansion, expansion, regular)

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    return jax.lax.igamma(self._concentration, self._rate * value)

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._concentration / self._rate

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return jnp.sqrt(self._concentration) / self._rate

  def variance(self) -> Array:
    """Calculates the variance."""
    return self._concentration / jnp.square(self._rate)

  def mode(self) -> Array:
    """Calculates the mode."""
    mode = (self._concentration - 1.0) / self._rate
    return jnp.where(self._concentration >= 1.0, mode, jnp.nan)

  def __getitem__(self, index) -> 'Gamma':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return Gamma(
        # pyrefly: ignore[bad-index]
        concentration=self.concentration[index], rate=self.rate[index])


def _kl_divergence_gamma_gamma(
    dist1: Union[Gamma, tfd.Gamma],
    dist2: Union[Gamma, tfd.Gamma],
    *unused_args,
    **unused_kwargs,
) -> Array:
  """Batched KL divergence KL(dist1 || dist2) between two Gamma distributions.

  Args:
    dist1: A Gamma distribution.
    dist2: A Gamma distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  t1 = dist2.concentration * (jnp.log(dist1.rate) - jnp.log(dist2.rate))
  t2 = jax.lax.lgamma(dist2.concentration) - jax.lax.lgamma(dist1.concentration)
  # pyrefly: ignore[unsupported-operation]
  t3 = (dist1.concentration - dist2.concentration) * jax.lax.digamma(
      dist1.concentration)
  # pyrefly: ignore[unsupported-operation]
  t4 = (dist2.rate - dist1.rate) * (dist1.concentration / dist1.rate)
  return t1 + t2 + t3 + t4


# Register the KL functions with TFP.
tfd.RegisterKL(Gamma, Gamma)(_kl_divergence_gamma_gamma)
tfd.RegisterKL(Gamma, Gamma.equiv_tfp_cls)(_kl_divergence_gamma_gamma)
tfd.RegisterKL(Gamma.equiv_tfp_cls, Gamma)(_kl_divergence_gamma_gamma)
