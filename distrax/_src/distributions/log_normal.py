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
"""Log-Normal distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions import normal as normal_lib
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class LogNormal(distribution.Distribution):
  """Log-Normal distribution on the positive real line (0, ∞).

  If X ~ LogNormal(μ, σ), then log(X) ~ Normal(μ, σ).

  **PDF**

      f(x; μ, σ) = 1 / (x σ √(2π)) * exp(-(log x − μ)² / (2σ²)),  x > 0

  **Properties**

  - Mean:     exp(μ + σ²/2)
  - Variance: (exp(σ²) − 1) * exp(2μ + σ²)
  - Median:   exp(μ)
  - Mode:     exp(μ − σ²)
  - Entropy:  log(σ * exp(μ + ½) * √(2π))
              = μ + ½ + log(σ) + ½ log(2π)
  """

  equiv_tfp_cls = tfd.LogNormal

  def __init__(self, loc: Numeric, scale: Numeric):
    """Initializes a LogNormal distribution.

    Args:
      loc:   Mean of the underlying Normal (μ in log-space).
      scale: Standard deviation of the underlying Normal (σ > 0).
    """
    super().__init__()
    self._loc   = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)
    self._batch_shape = jax.lax.broadcast_shapes(
        self._loc.shape, self._scale.shape)
    self._normal = normal_lib.Normal(loc=self._loc, scale=self._scale)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    return self._batch_shape

  @property
  def loc(self) -> Array:
    """Log-space mean μ."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Log-space standard deviation σ."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """See `Distribution._sample_n`."""
    # Sample Normal, then exponentiate.
    normal_samples = self._normal._sample_n(key, n)
    return jnp.exp(normal_samples)

  def log_prob(self, value: Array) -> Array:
    """See `Distribution.log_prob`.

    log f(x) = Normal(μ, σ).log_prob(log x) − log x
    """
    x = jnp.asarray(value, dtype=jnp.result_type(self._loc, self._scale))
    return self._normal.log_prob(jnp.log(x)) - jnp.log(x)

  def mean(self) -> Array:
    """See `Distribution.mean`.  E[X] = exp(μ + σ²/2)."""
    return jnp.exp(self.loc + 0.5 * jnp.square(self.scale))

  def variance(self) -> Array:
    """See `Distribution.variance`.
    Var[X] = (exp(σ²) − 1) * exp(2μ + σ²)
    """
    sigma2 = jnp.square(self.scale)
    return jnp.expm1(sigma2) * jnp.exp(2.0 * self.loc + sigma2)

  def mode(self) -> Array:
    """See `Distribution.mode`.  mode = exp(μ − σ²)."""
    return jnp.exp(self.loc - jnp.square(self.scale))

  def median(self) -> Array:
    """See `Distribution.median`.  median = exp(μ)."""
    return jnp.exp(self.loc)

  def entropy(self) -> Array:
    """See `Distribution.entropy`.

    H = μ + ½ + log(σ) + ½ log(2π)
    """
    return (self.loc
            + 0.5
            + jnp.log(self.scale)
            + 0.5 * jnp.log(2.0 * jnp.pi))

  def kl_divergence(self, other_dist, **kwargs) -> Array:
    """KL divergence KL(self ‖ other_dist).

    When ``other_dist`` is also :class:`LogNormal`, the analytic formula is:

    .. math::

        \\mathrm{KL}(\\text{LN}(\\mu_1, \\sigma_1) \\| \\text{LN}(\\mu_2, \\sigma_2))
        = \\log(\\sigma_2/\\sigma_1)
          + (\\sigma_1^2 + (\\mu_1-\\mu_2)^2) / (2\\sigma_2^2) - \\tfrac{1}{2}

    This equals :math:`\\mathrm{KL}(N(\\mu_1,\\sigma_1) \\| N(\\mu_2,\\sigma_2))`.
    """
    if isinstance(other_dist, LogNormal):
      return self._normal.kl_divergence(other_dist._normal)
    return super().kl_divergence(other_dist, **kwargs)
