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
"""Multinomial distribution."""

import functools
import operator

from typing import Tuple, Optional, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math
import jax
from jax import lax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Multinomial(distribution.Distribution):
  """Multinomial distribution with parameter `probs`."""

  equiv_tfp_cls = tfd.Multinomial

  def __init__(self,
               total_count: Numeric,
               logits: Optional[Array] = None,
               probs: Optional[Array] = None,
               dtype: jnp.dtype = int):
    """Initializes a Multinomial distribution.

    Args:
      total_count: The number of trials per sample.
      logits: Logit transform of the probability of each category. Only one
        of `logits` or `probs` can be specified.
      probs: Probability of each category. Only one of `logits` or `probs` can
        be specified.
      dtype:  The type of event samples.
    """
    super().__init__()
    logits = None if logits is None else conversion.as_float_array(logits)
    probs = None if probs is None else conversion.as_float_array(probs)
    if (logits is None) == (probs is None):
      raise ValueError(
          f'One and exactly one of `logits` and `probs` should be `None`, '
          f'but `logits` is {logits} and `probs` is {probs}.')
    if logits is not None and (not logits.shape or logits.shape[-1] < 2):
      raise ValueError(
          f'The last dimension of `logits` must be greater than 1, but '
          f'`logits.shape = {logits.shape}`.')
    if probs is not None and (not probs.shape or probs.shape[-1] < 2):
      raise ValueError(
          f'The last dimension of `probs` must be greater than 1, but '
          f'`probs.shape = {probs.shape}`.')
    if not (jnp.issubdtype(dtype, jnp.integer) or
            jnp.issubdtype(dtype, jnp.floating)):
      raise ValueError(
          f'The dtype of `{self.name}` must be integer or floating-point, '
          f'instead got `{dtype}`.')

    self._total_count = jnp.asarray(total_count, dtype=dtype)
    self._probs = None if probs is None else math.normalize(probs=probs)
    self._logits = None if logits is None else math.normalize(logits=logits)
    self._dtype = dtype

    if self._probs is not None:
      probs_batch_shape = self._probs.shape[:-1]
    else:
      assert self._logits is not None
      probs_batch_shape = self._logits.shape[:-1]
    self._batch_shape = lax.broadcast_shapes(
        probs_batch_shape, self._total_count.shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    if self._logits is not None:
      return self._logits.shape[-1:]
    else:
      return self._probs.shape[-1:]

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return self._batch_shape

  @property
  def total_count(self) -> Array:
    """The number of trials per sample."""
    return jnp.broadcast_to(self._total_count, self.batch_shape)

  @property
  def num_trials(self) -> Array:
    """The number of trials for each event."""
    return self.total_count

  @property
  def logits(self) -> Array:
    """The logits for each event."""
    if self._logits is not None:
      return jnp.broadcast_to(self._logits, self.batch_shape + self.event_shape)
    return jnp.broadcast_to(jnp.log(self._probs),
                            self.batch_shape + self.event_shape)

  @property
  def probs(self) -> Array:
    """The probabilities for each event."""
    if self._probs is not None:
      return jnp.broadcast_to(self._probs, self.batch_shape + self.event_shape)
    return jnp.broadcast_to(jax.nn.softmax(self._logits, axis=-1),
                            self.batch_shape + self.event_shape)

  @property
  def log_of_probs(self) -> Array:
    """The log probabilities for each event."""
    if self._logits is not None:
      # jax.nn.log_softmax was already applied in init to logits.
      return jnp.broadcast_to(self._logits,
                              self.batch_shape + self.event_shape)
    return jnp.broadcast_to(jnp.log(self._probs),
                            self.batch_shape + self.event_shape)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    total_permutations = lax.lgamma(self._total_count + 1.)
    counts_factorial = lax.lgamma(value + 1.)
    redundant_permutations = jnp.sum(counts_factorial, axis=-1)
    log_combinations = total_permutations - redundant_permutations
    return log_combinations + jnp.sum(
        math.multiply_no_nan(self.log_of_probs, value), axis=-1)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    num_keys = functools.reduce(operator.mul, self.batch_shape, 1)
    keys = jax.random.split(key, num=num_keys)
    total_count = jnp.reshape(self.total_count, (-1,))
    logits = jnp.reshape(self.logits, (-1,) + self.event_shape)
    sample_fn = jax.vmap(
        self._sample_n_scalar, in_axes=(0, 0, None, 0, None), out_axes=1)
    samples = sample_fn(keys, total_count, n, logits, self._dtype)  # [n, B, K]
    return samples.reshape((n,) + self.batch_shape + self.event_shape)

  @staticmethod
  def _sample_n_scalar(
      key: PRNGKey, total_count: int, n: int, logits: Array,
      dtype: jnp.dtype) -> Array:
    """Sample method for a Multinomial with integer `total_count`."""

    def cond_func(args):
      i, _, _ = args
      return jnp.less(i, total_count)

    def body_func(args):
      i, key_i, sample_aggregator = args
      key_i, current_key = jax.random.split(key_i)
      sample_i = jax.random.categorical(current_key, logits=logits, shape=(n,))
      one_hot_i = jax.nn.one_hot(sample_i, logits.shape[0]).astype(dtype)
      return i + 1, key_i, sample_aggregator + one_hot_i

    init_aggregator = jnp.zeros((n, logits.shape[0]), dtype=dtype)
    return lax.while_loop(cond_func, body_func, (0, key, init_aggregator))[2]

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    # The method `_entropy_scalar` does not work when `self.total_count` is an
    # array (instead of a scalar) or when we jit the function, so we default to
    # computing the entropy using an alternative method that uses a lax while
    # loop and does not create intermediate arrays whose shape depends on
    # `self.total_count`.
    entropy_fn = jnp.vectorize(
        self._entropy_scalar_with_lax, signature='(),(k),(k)->()')
    return entropy_fn(self.total_count, self.probs, self.log_of_probs)

  @staticmethod
  def _entropy_scalar(
      total_count: int, probs: Array, log_of_probs: Array
    ) -> Union[jnp.float32, jnp.float64]:
    """Calculates the entropy for a Multinomial with integer `total_count`."""
    # Constant factors in the entropy.
    xi = jnp.arange(total_count + 1, dtype=probs.dtype)
    log_xi_factorial = lax.lgamma(xi + 1)
    log_n_minus_xi_factorial = jnp.flip(log_xi_factorial, axis=-1)
    log_n_factorial = log_xi_factorial[..., -1]
    log_comb_n_xi = (
        log_n_factorial[..., None] - log_xi_factorial
        - log_n_minus_xi_factorial)
    comb_n_xi = jnp.round(jnp.exp(log_comb_n_xi))
    chex.assert_shape(comb_n_xi, (total_count + 1,))

    likelihood1 = math.power_no_nan(probs[..., None], xi)
    likelihood2 = math.power_no_nan(1. - probs[..., None], total_count - xi)
    chex.assert_shape(likelihood1, (probs.shape[-1], total_count + 1,))
    chex.assert_shape(likelihood2, (probs.shape[-1], total_count + 1,))
    likelihood = jnp.sum(likelihood1 * likelihood2, axis=-2)
    chex.assert_shape(likelihood, (total_count + 1,))
    comb_term = jnp.sum(comb_n_xi * log_xi_factorial * likelihood, axis=-1)
    chex.assert_shape(comb_term, ())

    # Probs factors in the entropy.
    n_probs_factor = jnp.sum(
        total_count * math.multiply_no_nan(log_of_probs, probs), axis=-1)

    return - log_n_factorial - n_probs_factor + comb_term

  @staticmethod
  def _entropy_scalar_with_lax(
      total_count: int, probs: Array, log_of_probs: Array
    ) -> Union[jnp.float32, jnp.float64]:
    """Like `_entropy_scalar`, but uses a lax while loop."""

    dtype = probs.dtype
    log_n_factorial = lax.lgamma(jnp.asarray(total_count + 1, dtype=dtype))

    def cond_func(args):
      xi, _ = args
      return jnp.less_equal(xi, total_count)

    def body_func(args):
      xi, accumulated_sum = args
      xi_float = jnp.asarray(xi, dtype=dtype)
      log_xi_factorial = lax.lgamma(xi_float + 1.)
      log_comb_n_xi = (log_n_factorial - log_xi_factorial
                       - lax.lgamma(total_count - xi_float + 1.))
      comb_n_xi = jnp.round(jnp.exp(log_comb_n_xi))
      likelihood1 = math.power_no_nan(probs, xi)
      likelihood2 = math.power_no_nan(1. - probs, total_count - xi)
      likelihood = likelihood1 * likelihood2
      comb_term = comb_n_xi * log_xi_factorial * likelihood  # [K]
      chex.assert_shape(comb_term, (probs.shape[-1],))
      return xi + 1, accumulated_sum + comb_term

    comb_term = jnp.sum(
        lax.while_loop(cond_func, body_func, (0, jnp.zeros_like(probs)))[1],
        axis=-1)

    n_probs_factor = jnp.sum(
        total_count * math.multiply_no_nan(log_of_probs, probs), axis=-1)

    return - log_n_factorial - n_probs_factor + comb_term

  def mean(self) -> Array:
    """Calculates the mean."""
    return self._total_count[..., None] * self.probs

  def variance(self) -> Array:
    """Calculates the variance."""
    probs = self.probs
    return self._total_count[..., None] * probs * (1. - probs)

  def covariance(self) -> Array:
    """Calculates the covariance."""
    probs = self.probs
    cov_matrix = -self._total_count[..., None, None] * (
        probs[..., None, :] * probs[..., :, None])
    chex.assert_shape(cov_matrix, probs.shape + self.event_shape)
    # Missing diagonal term in the covariance matrix.
    cov_matrix += jnp.vectorize(
        jnp.diag, signature='(k)->(k,k)')(
            self._total_count[..., None] * probs)
    return cov_matrix

  def __getitem__(self, index) -> 'Multinomial':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    total_count = self.total_count[index]
    if self._logits is not None:
      return Multinomial(
          total_count=total_count, logits=self.logits[index], dtype=self._dtype)
    return Multinomial(
        total_count=total_count, probs=self.probs[index], dtype=self._dtype)
