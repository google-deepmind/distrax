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
from distrax._src.utils import math
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class BetaBinomial(distribution.Distribution):
  """Beta-Binomial compound distribution.

  The Beta-Binomial distribution is parameterized by `total_count`,
  `concentration1` (alpha), and `concentration0` (beta).
  It is a compound distribution, equivalent to sampling a probability `p`
  from a `Beta(concentration1, concentration0)` distribution, and then
  sampling a count from a `Binomial(total_count, p)` distribution.

  The probability mass function (pmf) is,

  ```none
  pmf(k; n, a, b) = C(n, k) * Beta(k + a, n - k + b) / Beta(a, b)
  ```

  where:
  * `total_count = n`
  * `concentration1 = a > 0`
  * `concentration0 = b > 0`
  * `k` is the number of successes (an integer from 0 to n)
  * `C(n, k)` is the binomial coefficient "n choose k".
  * `Beta(x, y)` is the Beta function.
  """

  equiv_tfp_cls = tfd.BetaBinomial

  def __init__(
      self,
      total_count: Numeric,
      concentration1: Numeric,
      concentration0: Numeric,
      dtype: Union[jnp.dtype, type[Any]] = int,
  ):
    """Initializes a BetaBinomial distribution.

    Args:
      total_count: Non-negative floating-point tensor, whose components should
        be equal to integer values. The number of trials.
      concentration1: Positive floating-point tensor, the alpha parameter of the
        Beta prior.
      concentration0: Positive floating-point tensor, the beta parameter of the
        Beta prior.
      dtype: The type of event samples. Defaults to `int`.
    """
    super().__init__()
    # TFP implementation uses float for total_count, as do lbeta and
    # tfd.Binomial.
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
    return self._total_count

  @property
  def concentration1(self) -> Array:
    """Concentration parameter associated with a `success` outcome (alpha)."""
    return self._concentration1

  @property
  def concentration0(self) -> Array:
    """Concentration parameter associated with a `failure` outcome (beta)."""
    return self._concentration0

  def _log_combinations(self, n: Array, k: Array) -> Array:
    """Computes log(C(n, k)) using lbeta."""
    # log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
    #              = log(Gamma(n+1)) - log(Gamma(k+1)) - log(Gamma(n-k+1))
    # Using lbeta: lbeta(x, y) = log(Gamma(x)) + log(Gamma(y)) - log(Gamma(x+y))
    # Let x = k + 1, y = n - k + 1.
    # lbeta(k+1, n-k+1) = log(Gamma(k+1)) + log(Gamma(n-k+1)) - log(Gamma(n+2))
    # We want: log(Gamma(n+1)) - (log(Gamma(k+1)) + log(Gamma(n-k+1)))
    # This is: -lbeta(k + 1, n - k + 1) - log(n + 1)
    # log(C(n, k)) = -lbeta(k + 1, n - k + 1) - log(n + 1)
    # Using lbeta from distrax.utils.math
    return -math.log_beta(k + 1.0, n - k + 1.0) - jnp.log(n + 1.0)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    key1, key2, key3 = jax.random.split(key, 3)

    # Get parameters and broadcast them to (n,) + batch_shape
    shape = (n,) + self.batch_shape
    total_count = jnp.broadcast_to(self.total_count, shape)
    concentration1 = jnp.broadcast_to(self.concentration1, shape)
    concentration0 = jnp.broadcast_to(self.concentration0, shape)

    # Sample probs ~ Beta(concentration1, concentration0)
    # This is done by sampling g1 ~ Gamma(c1, 1) and g2 ~ Gamma(c0, 1)
    # and computing probs = g1 / (g1 + g2).
    g1 = jax.random.gamma(key1, concentration1)
    g2 = jax.random.gamma(key2, concentration0)

    g_sum = g1 + g2
    # Use 0.5 if g_sum is 0 (which happens if c1=0, c2=0), otherwise g1 / g_sum.
    probs = jnp.where(g_sum == 0.0, 0.5, g1 / g_sum)

    # Sample counts ~ Binomial(total_count, probs)
    samples = tfd.Binomial(total_count=total_count, probs=probs).sample(
        seed=key3
    )

    return samples.astype(self._dtype)

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    n = self.total_count
    # Cast value to jnp.asarray to ensure it has .astype method for pytype
    k = jnp.asarray(value).astype(n.dtype)
    c1 = self.concentration1
    c0 = self.concentration0

    # pmf(k; n, a, b) = C(n, k) * Beta(k + a, n - k + b) / Beta(a, b)
    # log_pmf = log(C(n, k)) + log(Beta(k + a, n - k + b)) - log(Beta(a, b))
    # log(Beta(x, y)) = log_beta(x, y)
    log_comb = self._log_combinations(n, k)
    log_beta_comp = math.log_beta(c1 + k, n - k + c0)
    log_beta_prior = math.log_beta(c1, c0)

    return log_comb + log_beta_comp - log_beta_prior

  def mean(self) -> Array:
    """See `Distribution.mean`."""
    n = self.total_count
    c1 = self.concentration1
    c0 = self.concentration0
    return n * c1 / (c1 + c0)

  def variance(self) -> Array:
    """See `Distribution.variance`."""
    n = self.total_count
    c1 = self.concentration1
    c0 = self.concentration0
    c_sum = c1 + c0
    # Formula from TFP:
    return (n * c1 * c0 * (c_sum + n)) / (c_sum**2 * (c_sum + 1.0))

  def __getitem__(self, index) -> 'BetaBinomial':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return BetaBinomial(
        total_count=self.total_count[index],
        concentration1=self.concentration1[index],
        concentration0=self.concentration0[index],
        dtype=self._dtype,
    )
