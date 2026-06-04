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

from typing import Any, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class Poisson(distribution.Distribution):
  """Poisson distribution with rate parameter λ.

  The Poisson distribution models the number of events occurring in a fixed
  interval when events occur independently at a constant rate λ > 0.

    P(X = k) = λ^k * exp(-λ) / k!,   k = 0, 1, 2, …

  **Properties**

  - Mean:     λ
  - Variance: λ
  - Mode:     floor(λ)   (λ − 1 if λ is a positive integer)
  - Entropy:  λ(1 − log λ) + exp(−λ) Σ_{k=0}^∞ λ^k log(k!) / k!
              (computed numerically by matching TFP)
  """

  equiv_tfp_cls = tfd.Poisson

  def __init__(
      self,
      rate: Numeric,
      dtype: Union[jnp.dtype, type[Any]] = int,
  ):
    """Initializes a Poisson distribution.

    Args:
      rate: Rate (λ) of the distribution.  Must be strictly positive.
      dtype: The type of event samples.  Must be integer or floating-point.
        Defaults to ``int``.
    """
    super().__init__()
    if not (jnp.issubdtype(dtype, jnp.integer) or
            jnp.issubdtype(dtype, jnp.floating)):
      raise ValueError(
          f'The dtype of `{self.name}` must be integer or floating-point, '
          f'instead got `{dtype}`.')
    self._rate = conversion.as_float_array(rate)
    self._dtype = dtype

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    return self._rate.shape

  @property
  def rate(self) -> Array:
    """The rate λ of the distribution."""
    return self._rate

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    out_shape = (n,) + self.batch_shape
    # jax.random.poisson requires an integer dtype internally.
    samples = jax.random.poisson(key, lam=self._rate, shape=out_shape,
                                 dtype=jnp.int32)
    return samples.astype(self._dtype)

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    value = jnp.asarray(value, dtype=jnp.float32)
    rate = jnp.broadcast_to(self._rate, self.batch_shape)
    # log P(k) = k log λ − λ − log Γ(k+1)
    return value * jnp.log(rate) - rate - jss.gammaln(value + 1)

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    return jnp.broadcast_to(self._rate, self.batch_shape)

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    return jnp.broadcast_to(self._rate, self.batch_shape)

  def mode(self) -> Array:
    """See `Distribution.mode`."""
    rate = jnp.broadcast_to(self._rate, self.batch_shape)
    return jnp.floor(rate).astype(self._dtype)

  def entropy(self) -> Array:
    """See `Distribution.entropy`.

    The entropy of the Poisson distribution has no closed-form expression.
    It is approximated by summing H = −Σ P(k) log P(k) over a window wide
    enough to capture essentially all probability mass.  The window is
    clipped to at most ``_MAX_ENTROPY_K`` to keep memory bounded; this is
    accurate for rates up to ~1000.
    """
    rate = jnp.broadcast_to(self._rate, self.batch_shape)
    # Upper bound: λ + 10√λ + 30 captures > 99.999 % mass; cap at 2000.
    k_max = jnp.minimum(
        jnp.ceil(rate + 10.0 * jnp.sqrt(rate) + 30.0).astype(jnp.int32).max(),
        jnp.array(2000, jnp.int32),
    )
    ks = jnp.arange(int(k_max) + 1, dtype=jnp.float32)  # (K,)
    # log P(k) = k log λ − λ − log Γ(k+1),  shape (batch, K)
    log_pk = (ks * jnp.log(rate[..., None])
              - rate[..., None]
              - jss.gammaln(ks + 1))
    # H = −Σ P(k) log P(k)  using the numerically stable mul_exp(log_p, log_p)
    from distrax._src.utils import math as dmath  # pylint: disable=g-import-not-at-top
    return -jnp.sum(dmath.mul_exp(log_pk, log_pk), axis=-1)

  def kl_divergence(self, other_dist, **kwargs) -> Array:
    """Calculates the KL divergence KL(self || other_dist).

    When ``other_dist`` is also a :class:`Poisson` distribution, the analytic
    formula is used:

    .. math::

        \\mathrm{KL}(\\mathrm{Poisson}(\\lambda_1) \\| \\mathrm{Poisson}(\\lambda_2))
        = \\lambda_1 \\log(\\lambda_1 / \\lambda_2) + \\lambda_2 - \\lambda_1

    Args:
      other_dist: The other distribution.
      **kwargs: Additional keyword arguments.

    Returns:
      KL divergence.
    """
    if isinstance(other_dist, Poisson):
      r1 = jnp.broadcast_to(self._rate, self.batch_shape)
      r2 = jnp.broadcast_to(other_dist.rate, other_dist.batch_shape)
      return r1 * (jnp.log(r1) - jnp.log(r2)) + r2 - r1
    return super().kl_divergence(other_dist, **kwargs)
