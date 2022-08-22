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
"""Categorical distribution."""

from typing import Optional, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey


class Categorical(distribution.Distribution):
  """Categorical distribution."""

  equiv_tfp_cls = tfd.Categorical

  def __init__(self,
               logits: Optional[Array] = None,
               probs: Optional[Array] = None,
               dtype: jnp.dtype = int):
    """Initializes a Categorical distribution.

    Args:
      logits: Logit transform of the probability of each category. Only one
        of `logits` or `probs` can be specified.
      probs: Probability of each category. Only one of `logits` or `probs` can
        be specified.
      dtype: The type of event samples.
    """
    super().__init__()
    if (logits is None) == (probs is None):
      raise ValueError(
          f'One and exactly one of `logits` and `probs` should be `None`, '
          f'but `logits` is {logits} and `probs` is {probs}.')
    if not (jnp.issubdtype(dtype, jnp.integer) or
            jnp.issubdtype(dtype, jnp.floating)):
      raise ValueError(
          f'The dtype of `{self.name}` must be integer or floating-point, '
          f'instead got `{dtype}`.')

    self._probs = None if probs is None else math.normalize(probs=probs)
    self._logits = None if logits is None else math.normalize(logits=logits)
    self._dtype = dtype

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def logits(self) -> Array:
    """The logits for each event."""
    if self._logits is not None:
      return self._logits
    return jnp.log(self._probs)

  @property
  def probs(self) -> Array:
    """The probabilities for each event."""
    if self._probs is not None:
      return self._probs
    return jax.nn.softmax(self._logits, axis=-1)

  @property
  def num_categories(self) -> int:
    """Number of categories."""
    if self._probs is not None:
      return self._probs.shape[-1]
    return self._logits.shape[-1]

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    new_shape = (n,) + self.logits.shape[:-1]
    is_valid = jnp.logical_and(jnp.all(jnp.isfinite(self.probs), axis=-1),
                               jnp.all(self.probs >= 0, axis=-1))
    draws = jax.random.categorical(key=key, logits=self.logits, axis=-1,
                                   shape=new_shape).astype(self._dtype)
    return jnp.where(is_valid, draws, jnp.ones_like(draws) * -1)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`."""
    value_one_hot = jax.nn.one_hot(value, self.num_categories)
    mask_outside_domain = jnp.logical_or(
        value < 0, value > self.num_categories - 1)
    return jnp.where(
        mask_outside_domain, -jnp.inf,
        jnp.sum(math.multiply_no_nan(self.logits, value_one_hot), axis=-1))

  def prob(self, value: Array) -> Array:
    """See `Distribution.prob`."""
    value_one_hot = jax.nn.one_hot(value, self.num_categories)
    return jnp.sum(math.multiply_no_nan(self.probs, value_one_hot), axis=-1)

  def entropy(self) -> Array:
    """See `Distribution.entropy`."""
    if self._logits is None:
      log_probs = jnp.log(self._probs)
    else:
      log_probs = jax.nn.log_softmax(self._logits)
    return -jnp.sum(math.mul_exp(log_probs, log_probs), axis=-1)

  def mode(self) -> Array:
    """See `Distribution.mode`."""
    parameter = self._probs if self._logits is None else self._logits
    return jnp.argmax(parameter, axis=-1).astype(self._dtype)

  def cdf(self, value: Array) -> Array:
    """See `Distribution.cdf`."""
    # For value < 0 the output should be zero because support = {0, ..., K-1}.
    should_be_zero = value < 0
    # For value >= K-1 the output should be one. Explicitly accounting for this
    # case addresses potential numerical issues that may arise when evaluating
    # derived methods (mainly, `log_survival_function`) for `value >= K-1`.
    should_be_one = value >= self.num_categories - 1
    # Will use value as an index below, so clip it to {0, ..., K-1}.
    value = jnp.clip(value, 0, self.num_categories - 1)
    value_one_hot = jax.nn.one_hot(value, self.num_categories)
    cdf = jnp.sum(math.multiply_no_nan(
        jnp.cumsum(self.probs, axis=-1), value_one_hot), axis=-1)
    return jnp.where(should_be_zero, 0., jnp.where(should_be_one, 1., cdf))

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`."""
    return jnp.log(self.cdf(value))

  def logits_parameter(self) -> Array:
    """Wrapper for `logits` property, for TFP API compatibility."""
    return self.logits

  def __getitem__(self, index) -> 'Categorical':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    if self._logits is not None:
      return Categorical(logits=self.logits[index], dtype=self._dtype)
    return Categorical(probs=self.probs[index], dtype=self._dtype)


CategoricalLike = Union[Categorical, tfd.Categorical]


def _kl_divergence_categorical_categorical(
    dist1: CategoricalLike,
    dist2: CategoricalLike,
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Obtains the KL divergence `KL(dist1 || dist2)` between two Categoricals.

  The KL computation takes into account that `0 * log(0) = 0`; therefore,
  `dist1` may have zeros in its probability vector.

  Args:
    dist1: A Categorical distribution.
    dist2: A Categorical distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.

  Raises:
    ValueError if the two distributions have different number of categories.
  """
  logits1 = dist1.logits_parameter()
  logits2 = dist2.logits_parameter()
  num_categories1 = logits1.shape[-1]
  num_categories2 = logits2.shape[-1]

  if num_categories1 != num_categories2:
    raise ValueError(
        f'Cannot obtain the KL between two Categorical distributions '
        f'with different number of categories: the first distribution has '
        f'{num_categories1} categories, while the second distribution has '
        f'{num_categories2} categories.')

  log_probs1 = jax.nn.log_softmax(logits1, axis=-1)
  log_probs2 = jax.nn.log_softmax(logits2, axis=-1)
  return jnp.sum(
      math.mul_exp(log_probs1 - log_probs2, log_probs1), axis=-1)


# Register the KL functions with TFP.
tfd.RegisterKL(Categorical, Categorical)(
    _kl_divergence_categorical_categorical)
tfd.RegisterKL(Categorical, Categorical.equiv_tfp_cls)(
    _kl_divergence_categorical_categorical)
tfd.RegisterKL(Categorical.equiv_tfp_cls, Categorical)(
    _kl_divergence_categorical_categorical)

# Also register the KL with the TFP OneHotCategorical.
tfd.RegisterKL(Categorical, tfd.OneHotCategorical)(
    _kl_divergence_categorical_categorical)
tfd.RegisterKL(tfd.OneHotCategorical, Categorical)(
    _kl_divergence_categorical_categorical)
