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
  """Diagonal Normal that accepts a `loc` and `log_scale` parameter.

  It is safe to use this class with negative values in `log_scale`, which will
  be exp'd and passed as `scale` into the standard Normal distribution.
  """

  def __init__(self,
               loc: Numeric,
               log_scale: Numeric,
               max_scale: Optional[float] = None):
    """Initializes a LogStddevNormal distribution.

    Args:
      loc: Mean of the distribution.
      log_scale: Log of the distribution's scale. This is often the
        pre-activated output of a neural network.
      max_scale: Maximum value of the scale that this distribution will saturate
        at. This parameter can be useful for numerical stability. It is not a
        hard maximum; rather, we compute scale as per the following formula:
        log(max_scale) - softplus(log(max_scale) - log_scale).
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
  def log_scale(self):
    """Distribution parameter for log standard deviation."""
    return jnp.broadcast_to(self._log_scale, self.batch_shape)

  def __getitem__(self, index) -> 'LogStddevNormal':
    """See `Distribution.__getitem__`."""
    index = distribution.to_batch_shape_index(self.batch_shape, index)
    return LogStddevNormal(
        loc=self.loc[index],
        log_scale=self.log_scale[index],
        max_scale=self._max_scale)


def _kl_logstddevnormal_logstddevnormal(p, q, *unused_args, **unused_kwargs):
  """Calculate the batched KL divergence KL(p || q) with p and q Normal.

  Args:
    p: A LogStddevNormal distribution.
    q: A LogStddevNormal distribution.

  Returns:
    Batchwise KL(p || q).
  """
  # KL[N(u_a, s_a^2) || N(u_b, s_b^2)] between two Gaussians.
  # (s_a^2 + (u_a - u_b)^2)/(2*s_b^2) + log(s_b) - log(s_a) - 1/2
  p_variance = jnp.square(p.scale)
  q_variance = jnp.square(q.scale)
  return ((p_variance + jnp.square(p.loc - q.loc))/(2.0 * q_variance) +
          q.log_scale - p.log_scale - 0.5)


# Register the KL function.
tfd.RegisterKL(LogStddevNormal, LogStddevNormal)(
    _kl_logstddevnormal_logstddevnormal)
