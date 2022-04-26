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
"""Bernoulli distribution."""

from typing import Optional, Tuple, Union

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


class Bernoulli(distribution.Distribution):
  """Bernoulli distribution.

  Bernoulli distribution with parameter `probs`, the probability of outcome `1`.
  """

  equiv_tfp_cls = tfd.Bernoulli

  def __init__(self,
               logits: Optional[Numeric] = None,
               probs: Optional[Numeric] = None,
               dtype: jnp.dtype = int):
    """Initializes a Bernoulli distribution.

    Args:
      logits: Logit transform of the probability of a `1` event (`0` otherwise),
        i.e. `probs = sigmoid(logits)`. Only one of `logits` or `probs` can be
        specified.
      probs: Probability of a `1` event (`0` otherwise). Only one of `logits` or
        `probs` can be specified.
      dtype: The type of event samples.
    """
    super().__init__()
    # Validate arguments.
    if (logits is None) == (probs is None):
      raise ValueError(
          f'One and exactly one of `logits` and `probs` should be `None`, '
          f'but `logits` is {logits} and `probs` is {probs}.')
    if not (jnp.issubdtype(dtype, bool) or
            jnp.issubdtype(dtype, jnp.integer) or
            jnp.issubdtype(dtype, jnp.floating)):
      raise ValueError(
          f'The dtype of `{self.name}` must be boolean, integer or '
          f'floating-point, instead got `{dtype}`.')
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
      return self._logits.shape
    return self._probs.shape

  @property
  def logits(self) -> Array:
    """The logits of a `1` event."""
    if self._logits is not None:
      return self._logits
    return jnp.log(self._probs) - jnp.log(1 - self._probs)

  @property
  def probs(self) -> Array:
    """The probabilities of a `1` event.."""
    if self._probs is not None:
      return self._probs
    return jax.nn.sigmoid(self._logits)

  def _log_probs_parameter(self) -> Tuple[Array, Array]:
    if self._logits is None:
      return (jnp.log1p(-1. * self._probs),
              jnp.log(self._probs))
    return (-jax.nn.softplus(self._logits),
            -jax.nn.softplus(-1. * self._logits))

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    probs = self.probs
    new_shape = (n,) + probs.shape
    uniform = jax.random.uniform(
        key=key, shape=new_shape, dtype=probs.dtype, minval=0., maxval=1.)
    return jnp.less(uniform, probs).astype(self._dtype)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    log_probs0, log_probs1 = self._log_probs_parameter()
    return (math.multiply_no_nan(log_probs0, 1 - value) +
            math.multiply_no_nan(log_probs1, value))

  def prob(self, value: Array) -> Array:
    """See `Distribution.prob`."""
    probs1 = self.probs
    probs0 = 1 - probs1
    return (math.multiply_no_nan(probs0, 1 - value) +
            math.multiply_no_nan(probs1, value))

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    # For value < 0 the output should be zero because support = {0, 1}.
    return jnp.where(value < 0,
                     jnp.array(0., dtype=self.probs.dtype),
                     jnp.where(value >= 1,
                               jnp.array(1.0, dtype=self.probs.dtype),
                               1 - self.probs))

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def entropy(self) -> Array:
    """See `Distribution.entropy`."""
    (probs0, probs1,
     log_probs0, log_probs1) = _probs_and_log_probs(self)
    return -1. * (
        math.multiply_no_nan(log_probs0, probs0) +
        math.multiply_no_nan(log_probs1, probs1))

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    return self.probs

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    return (1 - self.probs) * self.probs

  def mode(self) -> Array:
    """See `Distribution.probs`."""
    return (self.probs > 0.5).astype(self._dtype)

  def __getitem__(self, index) -> 'Bernoulli':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    if self._logits is not None:
      return Bernoulli(logits=self.logits[index], dtype=self._dtype)
    return Bernoulli(probs=self.probs[index], dtype=self._dtype)


def _probs_and_log_probs(
    dist: Union[Bernoulli, tfd.Bernoulli]
    ) -> Tuple[Array, Array, Array, Array]:
  """Calculates both `probs` and `log_probs`."""
  # pylint: disable=protected-access
  if dist._logits is None:
    probs0 = 1 - dist._probs
    probs1 = dist._probs
    log_probs0 = jnp.log1p(-1. * dist._probs)
    log_probs1 = jnp.log(dist._probs)
  else:
    probs0 = jax.nn.sigmoid(-1. * dist._logits)
    probs1 = jax.nn.sigmoid(dist._logits)
    log_probs0 = -jax.nn.softplus(dist._logits)
    log_probs1 = -jax.nn.softplus(-1. * dist._logits)
  return probs0, probs1, log_probs0, log_probs1


def _kl_divergence_bernoulli_bernoulli(
    dist1: Union[Bernoulli, tfd.Bernoulli],
    dist2: Union[Bernoulli, tfd.Bernoulli],
    *unused_args, **unused_kwargs,
    ) -> Array:
  """KL divergence `KL(dist1 || dist2)` between two Bernoulli distributions.

  Args:
    dist1: instance of a Bernoulli distribution.
    dist2: instance of a Bernoulli distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  one_minus_p1, p1, log_one_minus_p1, log_p1 = _probs_and_log_probs(dist1)
  _, _, log_one_minus_p2, log_p2 = _probs_and_log_probs(dist2)
  # KL[a || b] = Pa * Log[Pa / Pb] + (1 - Pa) * Log[(1 - Pa) / (1 - Pb)]
  # Multiply each factor individually to avoid Inf - Inf
  return (
      math.multiply_no_nan(log_p1, p1) -
      math.multiply_no_nan(log_p2, p1) +
      math.multiply_no_nan(log_one_minus_p1, one_minus_p1) -
      math.multiply_no_nan(log_one_minus_p2, one_minus_p1)
  )


# Register the KL functions with TFP.
tfd.RegisterKL(Bernoulli, Bernoulli)(
    _kl_divergence_bernoulli_bernoulli)
tfd.RegisterKL(Bernoulli, Bernoulli.equiv_tfp_cls)(
    _kl_divergence_bernoulli_bernoulli)
tfd.RegisterKL(Bernoulli.equiv_tfp_cls, Bernoulli)(
    _kl_divergence_bernoulli_bernoulli)
