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
"""BetaBinomial distribution."""

from typing import Any, Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import math as distrax_math
import jax
from jax import lax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class BetaBinomial(distribution.Distribution):
  """BetaBinomial distribution.

  The Beta-Binomial distribution is parameterized by `total_count`,
  `concentration1` (alpha), and `concentration0` (beta). It is a compound
  distribution, equivalent to sampling a probability `p` from a
  `Beta(concentration1, concentration0)` distribution, and then sampling `k`
  from a `Binomial(total_count, p)` distribution.
  """

  equiv_tfp_cls = tfd.BetaBinomial

  def __init__(
      self,
      total_count: Numeric,
      concentration1: Numeric,
      concentration0: Numeric,
      dtype: Union[jnp.dtype, type[Any]] = jnp.int_,
  ):
    """Initializes a BetaBinomial distribution.

    Args:
      total_count: Non-negative floating-point tensor, whose components should
        be equal to integer values. The number of trials.
      concentration1: Positive floating-point tensor, the alpha parameter of the
        underlying Beta distribution.
      concentration0: Positive floating-point tensor, the beta parameter of the
        underlying Beta distribution.
      dtype: The type of event samples.
    """
    super().__init__()
    if not (
        jnp.issubdtype(dtype, jnp.integer)
        or jnp.issubdtype(dtype, jnp.floating)
    ):
      raise ValueError(
          f'The dtype of `{self.name}` must be integer or '
          f'floating-point, instead got `{dtype}`.'
      )

    self._total_count = conversion.as_float_array(total_count)
    self._concentration1 = conversion.as_float_array(concentration1)
    self._concentration0 = conversion.as_float_array(concentration0)
    self._dtype = dtype

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    return jnp.broadcast_shapes(
        self._total_count.shape,
        self._concentration1.shape,
        self._concentration0.shape,
    )

  @property
  def total_count(self) -> Array:
    """Number of trials."""
    return jnp.broadcast_to(self._total_count, self.batch_shape)

  @property
  def concentration1(self) -> Array:
    """Concentration parameter alpha."""
    return jnp.broadcast_to(self._concentration1, self.batch_shape)

  @property
  def concentration0(self) -> Array:
    """Concentration parameter beta."""
    return jnp.broadcast_to(self._concentration0, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key_beta, key_binomial = jax.random.split(key)

    batch_shape = self.batch_shape
    sample_shape = (n,) + batch_shape

    # Sample probs from Beta
    probs = jax.random.beta(
        key_beta,
        a=jnp.broadcast_to(self._concentration1, sample_shape),
        b=jnp.broadcast_to(self._concentration0, sample_shape),
    )

    # Sample counts from Binomial
    total_count = jnp.broadcast_to(self._total_count, sample_shape)
    samples = jax.random.binomial(key_binomial, n=total_count, p=probs)

    return samples.astype(self._dtype)

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    k = conversion.as_float_array(value)
    n = self.total_count  # Use property to get broadcasted shape
    a = self.concentration1  # Use property to get broadcasted shape
    b = self.concentration0  # Use property to get broadcasted shape

    log_comb = (
        lax.lgamma(n + 1.0) - lax.lgamma(k + 1.0) - lax.lgamma(n - k + 1.0)
    )

    log_prob = (
        log_comb
        + distrax_math.log_beta(a + k, n - k + b)
        - distrax_math.log_beta(a, b)
    )

    return log_prob

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    return (
        self.total_count
        * self.concentration1
        / (self.concentration1 + self.concentration0)
    )

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    n = self.total_count
    a = self.concentration1
    b = self.concentration0
    c_sum = a + b
    return (n * a * b * (c_sum + n)) / (c_sum**2 * (c_sum + 1.0))

  def __getitem__(self, index) -> 'BetaBinomial':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return BetaBinomial(
        total_count=self.total_count[index],
        concentration1=self.concentration1[index],
        concentration0=self.concentration0[index],
        dtype=self._dtype,
    )
