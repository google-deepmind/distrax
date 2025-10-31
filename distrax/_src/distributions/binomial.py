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
"""Binomial distribution."""

from typing import Any, Optional, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class Binomial(distribution.Distribution):
  """Binomial distribution.

  Binomial distribution with parameters `total_count` (number of trials) and
  `probs` (probability of success for each trial).
  """

  equiv_tfp_cls = tfd.Binomial

  def __init__(
      self,
      total_count: Numeric,
      logits: Optional[Numeric] = None,
      probs: Optional[Numeric] = None,
      dtype: Union[jnp.dtype, type[Any]] = int,
  ):
    """Initializes a Binomial distribution.

    Args:
      total_count: Number of trials. Must be a non-negative integer.
      logits: Logit transform of the probability of success, i.e. `probs =
        sigmoid(logits)`. Only one of `logits` or `probs` can be specified.
      probs: Probability of success. Only one of `logits` or `probs` can be
        specified.
      dtype: The type of event samples.
    """
    super().__init__()
    # Validate arguments.
    if (logits is None) == (probs is None):
      raise ValueError(
          'One and exactly one of `logits` and `probs` should be `None`, '
          f'but `logits` is {logits} and `probs` is {probs}.'
      )
    if not (
        jnp.issubdtype(dtype, bool)
        or jnp.issubdtype(dtype, jnp.integer)
        or jnp.issubdtype(dtype, jnp.floating)
    ):
      raise ValueError(
          f'The dtype of `{self.name}` must be boolean, integer or '
          f'floating-point, instead got `{dtype}`.'
      )

    self._total_count = conversion.as_float_array(total_count)
    if not jnp.issubdtype(self._total_count.dtype, jnp.integer):
      # From TFP:
      # If `total_count` is float-like, then we require it to be integer-valued.
      # We check this here and not in the `tf.function` graph to allow
      # non-integer-valued floats to be passed to `log_prob`.
      if not jnp.all(self._total_count == jnp.floor(self._total_count)):
        raise ValueError(
            '`total_count` must be integer-valued, but got '
            f'{self._total_count}.'
        )
      self._total_count = self._total_count.astype(int)

    # Parameters of the distribution.
    self._probs = None if probs is None else conversion.as_float_array(probs)
    self._logits = None if logits is None else conversion.as_float_array(logits)
    self._dtype = dtype

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    if self._logits is not None:
      shape = self._logits.shape
    else:
      shape = self._probs.shape
    return jnp.broadcast_shapes(shape, self._total_count.shape)

  @property
  def logits(self) -> Array:
    """The logits of a `1` event."""
    if self._logits is not None:
      return self._logits
    return jnp.log(self._probs) - jnp.log(1 - self._probs)

  @property
  def probs(self) -> Array:
    """The probabilities of a `1` event."""
    if self._probs is not None:
      return self._probs
    return jax.nn.sigmoid(self._logits)

  @property
  def total_count(self) -> Array:
    """The number of trials."""
    return self._total_count

  def _log_probs_parameter(self) -> Tuple[Array, Array]:
    """Computes log-probabilities from logits or probs."""
    if self._logits is None:
      return (jnp.log1p(-1.0 * self._probs), jnp.log(self._probs))
    return (
        -jax.nn.softplus(self._logits),
        -jax.nn.softplus(-1.0 * self._logits),
    )

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    probs = self.probs
    total_count = self.total_count

    # Broadcast shapes.
    sample_shape = (n,) + self.batch_shape
    probs = jnp.broadcast_to(probs, sample_shape)
    total_count = jnp.broadcast_to(total_count, sample_shape)

    # Sample using tfp.random.binomial
    # Note: jax.random.binomial exists and is preferred.
    samples = random.binomial(key, n=total_count, p=probs, shape=sample_shape)
    return samples.astype(self._dtype)

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    value_arr = jnp.asarray(value)
    value_int = value_arr.astype(self._total_count.dtype)

    # Calculate log_prob component from success/failure
    log_probs0, log_probs1 = self._log_probs_parameter()
    log_unnormalized = math.multiply_no_nan(
        value_int, log_probs1
    ) + math.multiply_no_nan(self._total_count - value_int, log_probs0)

    # Calculate log-normalization (log combinations)
    log_normalization = (
        lax.lgamma(self._total_count + 1.0)
        - lax.lgamma(value_int + 1.0)
        - lax.lgamma(self._total_count - value_int + 1.0)
    )

    # Set log_prob to -inf for invalid values (k > n or k < 0)
    # The lgamma functions will return +inf for non-positive integer args,
    # which makes log_normalization = -inf, and log_prob = -inf.
    # This correctly handles k < 0 and k > n.
    # We only need to manually handle non-integer values.
    valid_values = value_int == value_arr
    log_prob = log_unnormalized + log_normalization

    return jnp.where(valid_values, log_prob, -jnp.inf)

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    k = jnp.floor(value)
    n = self._total_count
    p = self.probs

    # Using the regularized incomplete beta function (like TFP)
    # cdf(k; n, p) = I_{1-p}(n - k, k + 1)

    a = n - k
    b = k + 1

    # Replicate TFP's safeguard for `a`.
    ones = jnp.ones_like(a)
    safe_a = jnp.where(jnp.logical_or(k < 0, k >= n), ones, a)

    # Replicate TFP's safeguard for `b`.
    # `lax.betainc` requires b > 0.
    # We set `b` to 1.0 where it would be <= 0.
    # The result is masked out by the `jnp.where(k < 0, 0.0, ...)` below.
    safe_b = jnp.where(b <= 0, ones, b)

    x = 1.0 - p

    # Compute the `betainc` with safe arguments.
    cdf_val = lax.betainc(safe_a, safe_b, x)

    # Handle values outside support, using `k` (the floor)
    # This matches TFP's `extend_cdf_outside_support`
    cdf_val = jnp.where(k < 0, 0.0, cdf_val)
    cdf_val = jnp.where(k >= n, 1.0, cdf_val)
    return cdf_val

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def entropy(self) -> Array:
    """See `Distribution.entropy`."""
    # Entropy for Binomial is not trivial.
    # TFP uses a bespoke kernel `binomial_entropy`.
    # We delegate to the equivalent TFP distribution.
    return self.equiv_tfp_cls(
        total_count=self._total_count, probs=self.probs
    ).entropy()

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    return self._total_count * self.probs

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    return self._total_count * self.probs * (1.0 - self.probs)

  def mode(self) -> Array:
    """See `Distribution.mode`."""
    return jnp.floor((self._total_count + 1.0) * self.probs).astype(self._dtype)

  def __getitem__(self, index) -> 'Binomial':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    if self._logits is not None:
      return Binomial(
          total_count=self.total_count[index],
          logits=self.logits[index],
          dtype=self._dtype,
      )
    return Binomial(
        total_count=self.total_count[index],
        probs=self.probs[index],
        dtype=self._dtype,
    )


def _probs_and_log_probs(
    dist: Union[Binomial, tfd.Binomial],
) -> Tuple[Array, Array, Array, Array]:
  """Calculates both `probs` and `log_probs` from dist params."""
  # pylint: disable=protected-access
  if dist._logits is None:
    probs0 = 1.0 - dist._probs
    probs1 = 1.0 - probs0
    log_probs0 = jnp.log1p(-1.0 * dist._probs)
    log_probs1 = jnp.log(dist._probs)
  else:
    probs0 = jax.nn.sigmoid(-1.0 * dist._logits)
    probs1 = jax.nn.sigmoid(dist._logits)
    log_probs0 = -jax.nn.softplus(dist._logits)
    log_probs1 = -jax.nn.softplus(-1.0 * dist._logits)
  return probs0, probs1, log_probs0, log_probs1


def _kl_divergence_binomial_binomial(
    dist1: Union[Binomial, tfd.Binomial],
    dist2: Union[Binomial, tfd.Binomial],
    *unused_args,
    **unused_kwargs,
) -> Array:
  """KL divergence `KL(dist1 || dist2)` between two Binomial distributions.

  Args:
    dist1: instance of a Binomial distribution.
    dist2: instance of a Binomial distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  # This KL is only defined when dist1.total_count == dist2.total_count.
  # TFP implementation asserts this. We assume it holds.
  # KL[a || b] = E_a[log(p_a(X) / p_b(X))]
  #            = E_a[log_comb(n, X) + X*log(p1) + (n-X)*log(1-p1)]
  #            - E_a[log_comb(n, X) + X*log(p2) + (n-X)*log(1-p2)]
  #            = E_a[X*log(p1) + (n-X)*log(1-p1) - X*log(p2) - (n-X)*log(1-p2)]
  #            = E_a[X * (log(p1) - log(p2)) + (n-X) * (log(1-p1) - log(1-p2))]
  #            = E[X] * (log(p1) - log(p2)) + (n-E[X]) * (log(1-p1) - log(1-p2))
  #            = n*p1 * (log(p1) - log(p2)) + (n-n*p1) * (log(1-p1) - log(1-p2))
  #            = n*p1 * (log_p1 - log_p2) + n*(1-p1) * (log_1_p1 - log_1_p2)

  # pylint: disable=protected-access
  n1 = dist1._total_count
  _, p1, log_1_p1, log_p1 = _probs_and_log_probs(dist1)
  _, _, log_1_p2, log_p2 = _probs_and_log_probs(dist2)

  kl = n1 * p1 * (log_p1 - log_p2) + n1 * (1.0 - p1) * (log_1_p1 - log_1_p2)

  # Following TFP, we should check n1 == n2.
  # Since we can't assert, we can return `inf` where they are not equal.
  n2 = dist2._total_count
  n1_b, n2_b = jnp.broadcast_arrays(n1, n2)
  return jnp.where(n1_b == n2_b, kl, jnp.inf)


# Register the KL functions with TFP.
tfd.RegisterKL(Binomial, Binomial)(_kl_divergence_binomial_binomial)
tfd.RegisterKL(Binomial, Binomial.equiv_tfp_cls)(
    _kl_divergence_binomial_binomial
)
tfd.RegisterKL(Binomial.equiv_tfp_cls, Binomial)(
    _kl_divergence_binomial_binomial
)
