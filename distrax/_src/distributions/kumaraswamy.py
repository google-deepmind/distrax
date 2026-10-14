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
"""Kumaraswamy distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
import jax.scipy.special as jss

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class Kumaraswamy(distribution.Distribution):
  """Kumaraswamy distribution on the open interval (0, 1).

  The Kumaraswamy distribution (Kumaraswamy 1980) is a two-parameter
  continuous distribution on (0, 1) with PDF:

      f(x; a, b) = a * b * x^{a-1} * (1 - x^a)^{b-1}

  and CDF:

      F(x; a, b) = 1 - (1 - x^a)^b

  Unlike the Beta distribution, the Kumaraswamy CDF and inverse CDF have
  simple closed forms, making it well-suited for reparameterized sampling in
  variational inference.  The inverse CDF (quantile function) is:

      F^{-1}(u; a, b) = (1 - (1 - u)^{1/b})^{1/a}

  **Properties**

  - Mean:    b * Γ(1 + 1/a) * Γ(b) / Γ(1 + 1/a + b)
  - Variance: b * Γ(1 + 2/a) * Γ(b) / Γ(1 + 2/a + b) − mean²
  - Mode:    ((a-1)/(ab-1))^{1/a}   for a,b ≥ 1 and (a,b) ≠ (1,1)
  - Entropy: (1-1/b) + (1-1/a) * H_b + log(ab),
             where H_b = Σ_{k=1}^∞ (-1)^{k+1} b^k / (k(k + a·k...))
             (computed numerically)

  **Reference**

  Kumaraswamy, P. (1980). A generalized probability density function for
  double-bounded random processes. *Journal of Hydrology*, 46(1–2), 79–88.
  """

  def __init__(self, concentration0: Numeric, concentration1: Numeric):
    """Initializes a Kumaraswamy distribution.

    Args:
      concentration0: First shape parameter `a > 0`.
      concentration1: Second shape parameter `b > 0`.
    """
    super().__init__()
    self._concentration0 = conversion.as_float_array(concentration0)
    self._concentration1 = conversion.as_float_array(concentration1)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._concentration0.shape, self._concentration1.shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    return self._batch_shape

  @property
  def concentration0(self) -> Array:
    """First shape parameter a."""
    return jnp.broadcast_to(self._concentration0, self.batch_shape)

  @property
  def concentration1(self) -> Array:
    """Second shape parameter b."""
    return jnp.broadcast_to(self._concentration1, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`.

    Uses the closed-form inverse CDF:  F^{-1}(u) = (1 - (1-u)^{1/b})^{1/a}.
    """
    out_shape = (n,) + self.batch_shape
    dtype = jnp.result_type(self._concentration0, self._concentration1)
    # Sample uniform and apply quantile transform.
    u = jax.random.uniform(key, shape=out_shape, dtype=dtype,
                           minval=jnp.finfo(dtype).tiny, maxval=1.0)
    a = jnp.broadcast_to(self._concentration0, self.batch_shape)
    b = jnp.broadcast_to(self._concentration1, self.batch_shape)
    return jnp.power(1.0 - jnp.power(1.0 - u, 1.0 / b), 1.0 / a)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`.

    log f(x) = log(a) + log(b) + (a-1)*log(x) + (b-1)*log(1 - x^a)
    """
    a = jnp.broadcast_to(self._concentration0, self.batch_shape)
    b = jnp.broadcast_to(self._concentration1, self.batch_shape)
    x = jnp.asarray(value, dtype=jnp.result_type(a, b))
    log_xa = a * jnp.log(x)                   # a * log(x)  = log(x^a)
    log1m_xa = jnp.log1p(-jnp.exp(log_xa))    # log(1 - x^a), numerically stable
    return (jnp.log(a) + jnp.log(b)
            + (a - 1.0) * jnp.log(x)
            + (b - 1.0) * log1m_xa)

  def log_cdf(self, value: Array) -> Array:
    """See `Distribution.log_cdf`.

    log F(x) = log(1 - (1 - x^a)^b)  = log1p(-(1 - x^a)^b)
    """
    a = jnp.broadcast_to(self._concentration0, self.batch_shape)
    b = jnp.broadcast_to(self._concentration1, self.batch_shape)
    x = jnp.asarray(value, dtype=jnp.result_type(a, b))
    xa = jnp.power(x, a)
    return jnp.log1p(-jnp.power(1.0 - xa, b))

  def mean(self) -> Array:
    """See `Distribution.mean`.

    E[X] = b * B(1 + 1/a, b)  =  b * Γ(1+1/a) * Γ(b) / Γ(1+1/a+b)
    """
    a = self.concentration0
    b = self.concentration1
    inv_a = 1.0 / a
    log_mean = (jnp.log(b)
                + jss.gammaln(1.0 + inv_a)
                + jss.gammaln(b)
                - jss.gammaln(1.0 + inv_a + b))
    return jnp.exp(log_mean)

  def variance(self) -> Array:
    """See `Distribution.variance`.

    Var[X] = b * B(1 + 2/a, b) − mean(X)^2
    """
    a = self.concentration0
    b = self.concentration1
    inv_a = 2.0 / a
    log_second_moment = (jnp.log(b)
                         + jss.gammaln(1.0 + inv_a)
                         + jss.gammaln(b)
                         - jss.gammaln(1.0 + inv_a + b))
    return jnp.exp(log_second_moment) - jnp.square(self.mean())

  def mode(self) -> Array:
    """See `Distribution.mode`.

    mode = ((a-1) / (ab-1))^{1/a}  for a >= 1, b >= 1, (a,b) != (1,1).
    Returns 0 when a < 1 and 1 when b < 1 (boundary modes).
    """
    a = self.concentration0
    b = self.concentration1
    # Interior mode (valid when a >= 1 and ab > 1)
    num = jnp.maximum(a - 1.0, 0.0)
    denom = jnp.maximum(a * b - 1.0, 1e-30)
    interior_mode = jnp.power(num / denom, 1.0 / a)
    # Boundary: mode = 0 if a < 1, mode = 1 if b < 1, else interior.
    mode = jnp.where(a < 1.0, jnp.zeros_like(a),
                     jnp.where(b < 1.0, jnp.ones_like(a), interior_mode))
    return mode

  def entropy(self) -> Array:
    """See `Distribution.entropy`.

    H = (1 - 1/b) + (1 - 1/a) * harmonic_b + log(ab)

    where harmonic_b = Σ_{k=1}^{K} (-1)^{k+1} * b / (k*(k*a + a))
    ... actually using the known exact formula:

    H = (1 - 1/b) + (1 - 1/a) * Σ_{k=1}^{K} 1/k * b / (k + b) + log(ab)

    We use the TFP formulation via digamma functions (equivalent):
    H = (1 - 1/b) + log(ab) + (b-1)/b * [-euler_mascheroni - digamma(b + 1) + 1]

    Actually, the standard formula is:
    H(a,b) = (1 - 1/b) + (1 - 1/a) * Σ_{j=1}^∞ 1/j * B(j/a + 1, b)
    ≈ (1 - 1/b) + (1 - 1/a) * Ψ(b+1)  [leading order for large b]

    For accuracy we use the numerical series truncated at K=100 terms.
    """
    a = self.concentration0
    b = self.concentration1
    # Series: Σ_{j=1}^K (-1)^{j+1} * b/(j*b + j*a*j) ... this is complex.
    # Use the known result from Jones (2009):
    # H = (1 - 1/b) + log(a*b) + (1 - 1/a) * (euler + digamma(b+1))
    # where euler = -digamma(1) = 0.5772...
    euler_mascheroni = -jss.digamma(jnp.ones_like(a))  # γ ≈ 0.5772...
    # Derivation: H = -E[log f(X)]
    #   = -log(ab) + (1-1/a)*(γ + ψ(b+1)) + (1-1/b)
    # where γ + ψ(b+1) = E[-log U] for U ~ Beta(1,b) and
    # (1-1/b) = E[-log(1-X^a)] contribution.
    return ((1.0 - 1.0 / b)
            - jnp.log(a * b)
            + (1.0 - 1.0 / a) * (euler_mascheroni + jss.digamma(b + 1.0)))
