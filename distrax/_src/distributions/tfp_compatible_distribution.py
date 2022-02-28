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
"""Wrapper to adapt a Distrax distribution for use in TFP."""

from typing import Dict, Optional, Sequence, Tuple, Union

import chex
from distrax._src.distributions import distribution
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
Distribution = distribution.Distribution
IntLike = distribution.IntLike
PRNGKey = chex.PRNGKey
tangent_spaces = tfp.experimental.tangent_spaces
TangentSpace = tangent_spaces.TangentSpace


def tfp_compatible_distribution(
    base_distribution: Distribution,
    name: Optional[str] = None) -> distribution.DistributionT:
  """Create a TFP-compatible distribution from a Distrax distribution.

  Given a Distrax distribution, return a wrapped distribution that behaves as a
  TFP distribution, to be used in TFP meta-distributions. In particular, the
  wrapped distribution implements the methods `allow_nan_stats`, `parameters`,
  `name`, `batch_shape_tensor`, `reparameterization_type` and
  `event_shape_tensor`; and the `batch_shape` and `event_shape` properties
  return a TFP `TensorShape`.

  The methods of the resulting distribution do not take a `name` argument,
  unlike their TFP equivalents.

  Args:
    base_distribution: A Distrax distribution.
    name: The distribution name.

  Returns:
    The wrapped distribution.
  """

  name_ = name

  class TFPCompatibleDistribution(base_distribution.__class__):
    """Class to wrap a Distrax distribution.

    The wrapped class dynamically inherits from `base_distribution`, so that
    computations involving the KL remain valid.
    """

    def __init__(self):
      pass

    def __getattr__(self, name: str):
      return getattr(base_distribution, name)

    def __getitem__(self, index):
      return tfp_compatible_distribution(base_distribution[index], name=name_)

    @property
    def allow_nan_stats(self) -> bool:
      """Proxy for the TFP property `allow_nan_stats`.

      It always returns True.
      """
      return True

    @property
    def batch_shape(self) -> tfp.tf2jax.TensorShape:
      """Returns a `TensorShape` with the `batch_shape` of the distribution."""
      return tfp.tf2jax.TensorShape(base_distribution.batch_shape)

    def batch_shape_tensor(self) -> Array:
      """See `Distribution.batch_shape`."""
      return jnp.array(base_distribution.batch_shape, dtype=jnp.int32)

    @property
    def event_shape(self) -> tfp.tf2jax.TensorShape:
      """Returns a `TensorShape` with the `event_shape` of the distribution."""
      return tfp.tf2jax.TensorShape(base_distribution.event_shape)

    def event_shape_tensor(self) -> ArrayNumpy:
      """See `Distribution.event_shape`."""
      return np.array(base_distribution.event_shape, dtype=jnp.int32)

    @property
    def experimental_shard_axis_names(self):
      return []

    @property
    def name(self) -> str:
      """See `Distribution.name`."""
      return name_ or f'TFPCompatible{base_distribution.name}'

    @property
    def reparameterization_type(self) -> tfd.ReparameterizationType:
      """Proxy for the TFP property `reparameterization_type`.

      It always returns `tfd.NOT_REPARAMETERIZED`.
      """
      return tfd.NOT_REPARAMETERIZED

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
      return base_distribution.sample(seed=key, sample_shape=(n,))

    def log_prob(self, value: Array) -> Array:
      """See `Distribution.log_prob`."""
      return base_distribution.log_prob(value)

    @property
    def parameters(self) -> Dict[str, str]:
      """Returns a dictionary whose key 'name' maps to the distribution name."""
      return {'name': self.name}

    def sample(self,
               sample_shape: Union[IntLike, Sequence[IntLike]] = (),
               seed: Optional[Union[int, tfp.util.SeedStream]] = None,
               **unused_kwargs) -> Array:
      """See `Distribution.sample`."""
      if not np.isscalar(sample_shape):
        sample_shape = tuple(sample_shape)
      return base_distribution.sample(sample_shape=sample_shape, seed=seed)

    def experimental_local_measure(
        self,
        value: Array,
        backward_compat: bool = True,
        **unused_kwargs) -> Tuple[Array, TangentSpace]:
      """Returns a log probability density together with a `TangentSpace`.

      See `tfd.distribution.Distribution.experimental_local_measure`, and
      Radul and Alexeev, AISTATS 2021, “The Base Measure Problem and its
      Solution”, https://arxiv.org/abs/2010.09647.

      Args:
        value: `float` or `double` `Array`.
        backward_compat: unused
        **unused_kwargs: unused

      Returns:
        log_prob: see `log_prob`.
        tangent_space: `tangent_spaces.FullSpace()`, representing R^n with the
          standard basis.
      """
      del backward_compat
      # We ignore the `backward_compat` flag and always act as though it's
      # true because Distrax bijectors and distributions need not follow the
      # base measure protocol from TFP.
      del unused_kwargs
      return self.log_prob(value), tangent_spaces.FullSpace()

  return TFPCompatibleDistribution()
