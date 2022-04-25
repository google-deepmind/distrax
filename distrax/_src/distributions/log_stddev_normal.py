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
"""LogStddevNormal distribution."""

import math
from typing import Optional

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions import normal
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric


class LogStddevNormal(normal.Normal):
  """Normal distribution with `log_scale` parameter.

  The `LogStddevNormal` has three parameters: `loc`, `log_scale`, and
  (optionally) `max_scale`. The distribution is a univariate normal
  distribution with mean equal to `loc` and scale parameter (i.e., stddev) equal
  to `exp(log_scale)` if `max_scale` is None. If `max_scale` is not None, a soft
  thresholding is applied to obtain the scale parameter of the normal, so that
  its log is given by `log(max_scale) - softplus(log(max_scale) - log_scale)`.
  """

  def __init__(self,
               loc: Numeric,
               log_scale: Numeric,
               max_scale: Optional[float] = None):
    """Initializes a LogStddevNormal distribution.

    Args:
      loc: Mean of the distribution.
      log_scale: Log of the distribution's scale (before the soft thresholding
        applied when `max_scale` is not None).
      max_scale: Maximum value of the scale that this distribution will saturate
        at. This parameter can be useful for numerical stability. It is not a
        hard maximum; rather, we compute `log(scale)` as per the formula:
        `log(max_scale) - softplus(log(max_scale) - log_scale)`.
    """
    self._max_scale = max_scale
    if max_scale is not None:
      max_log_scale = math.log(max_scale)
      self._log_scale = max_log_scale - jax.nn.softplus(
          max_log_scale - conversion.as_float_array(log_scale))
    else:
      self._log_scale = conversion.as_float_array(log_scale)
    scale = jnp.exp(self._log_scale)
    super().__init__(loc, scale)

  @property
  def log_scale(self) -> Array:
    """The log standard deviation (after thresholding, if applicable)."""
    return jnp.broadcast_to(self._log_scale, self.batch_shape)

  def __getitem__(self, index) -> 'LogStddevNormal':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return LogStddevNormal(
        loc=self.loc[index],
        log_scale=self.log_scale[index],
        max_scale=self._max_scale)


def _kl_logstddevnormal_logstddevnormal(
    dist1: LogStddevNormal, dist2: LogStddevNormal,
    *unused_args, **unused_kwargs) -> Array:
  """Calculates the batched KL divergence between two LogStddevNormal's.

  Args:
    dist1: A LogStddevNormal distribution.
    dist2: A LogStddevNormal distribution.

  Returns:
    Batchwise KL(dist1 || dist2).
  """
  # KL[N(u_a, s_a^2) || N(u_b, s_b^2)] between two Gaussians:
  # (s_a^2 + (u_a - u_b)^2)/(2*s_b^2) + log(s_b) - log(s_a) - 1/2.
  variance1 = jnp.square(dist1.scale)
  variance2 = jnp.square(dist2.scale)
  return ((variance1 + jnp.square(dist1.loc - dist2.loc)) / (2.0 * variance2) +
          dist2.log_scale - dist1.log_scale - 0.5)


# Register the KL function.
tfd.RegisterKL(LogStddevNormal, LogStddevNormal)(
    _kl_logstddevnormal_logstddevnormal)
