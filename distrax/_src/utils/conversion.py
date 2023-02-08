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
"""Utility functions for conversion between different types."""

from typing import Optional, Union

import chex
from distrax._src.bijectors import bijector
from distrax._src.bijectors import bijector_from_tfp
from distrax._src.bijectors import lambda_bijector
from distrax._src.bijectors import sigmoid
from distrax._src.bijectors import tanh
from distrax._src.bijectors import tfp_compatible_bijector
from distrax._src.distributions import distribution
from distrax._src.distributions import distribution_from_tfp
from distrax._src.distributions import tfp_compatible_distribution
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
BijectorLike = bijector.BijectorLike
DistributionLike = distribution.DistributionLike


def to_tfp(obj: Union[bijector.Bijector, tfb.Bijector,
                      distribution.Distribution, tfd.Distribution],
           name: Optional[str] = None):
  """Converts a distribution or bijector to a TFP-compatible equivalent object.

  The returned object is not necessarily of type `tfb.Bijector` or
  `tfd.Distribution`; rather, it is a Distrax object that implements TFP
  functionality so that it can be used in TFP.

  If the input is already of TFP type, it is returned unchanged.

  Args:
    obj: The distribution or bijector to be converted to TFP.
    name: The name of the resulting object.

  Returns:
    A TFP-compatible equivalent distribution or bijector.
  """
  if isinstance(obj, (tfb.Bijector, tfd.Distribution)):
    return obj
  elif isinstance(obj, bijector.Bijector):
    return tfp_compatible_bijector.tfp_compatible_bijector(obj, name)
  elif isinstance(obj, distribution.Distribution):
    return tfp_compatible_distribution.tfp_compatible_distribution(obj, name)
  else:
    raise TypeError(
        f"`to_tfp` can only convert objects of type: `distrax.Bijector`,"
        f" `tfb.Bijector`, `distrax.Distribution`, `tfd.Distribution`. Got type"
        f" `{type(obj)}`.")


def as_bijector(obj: BijectorLike) -> bijector.BijectorT:
  """Converts a bijector-like object to a Distrax bijector.

  Bijector-like objects are: Distrax bijectors, TFP bijectors, and callables.
  Distrax bijectors are returned unchanged. TFP bijectors are converted to a
  Distrax equivalent. Callables are wrapped by `distrax.Lambda`, with a few
  exceptions where an explicit implementation already exists and is returned.

  Args:
    obj: The bijector-like object to be converted.

  Returns:
    A Distrax bijector.
  """
  if isinstance(obj, bijector.Bijector):
    return obj
  elif isinstance(obj, tfb.Bijector):
    return bijector_from_tfp.BijectorFromTFP(obj)
  elif obj is jax.nn.sigmoid:
    return sigmoid.Sigmoid()
  elif obj is jnp.tanh:
    return tanh.Tanh()
  elif callable(obj):
    return lambda_bijector.Lambda(obj)
  else:
    raise TypeError(
        f"A bijector-like object can be a `distrax.Bijector`, a `tfb.Bijector`,"
        f" or a callable. Got type `{type(obj)}`.")


def as_distribution(obj: DistributionLike) -> distribution.DistributionT:
  """Converts a distribution-like object to a Distrax distribution.

  Distribution-like objects are: Distrax distributions and TFP distributions.
  Distrax distributions are returned unchanged. TFP distributions are converted
  to a Distrax equivalent.

  Args:
    obj: A distribution-like object to be converted.

  Returns:
    A Distrax distribution.
  """
  if isinstance(obj, distribution.Distribution):
    return obj
  elif isinstance(obj, tfd.Distribution):
    return distribution_from_tfp.distribution_from_tfp(obj)
  else:
    raise TypeError(
        f"A distribution-like object can be a `distrax.Distribution` or a"
        f" `tfd.Distribution`. Got type `{type(obj)}`.")


def as_float_array(x: Numeric) -> Array:
  """Converts input to an array with floating-point dtype.

  If the input is already an array with floating-point dtype, it is returned
  unchanged.

  Args:
    x: input to convert.

  Returns:
    An array with floating-point dtype.
  """
  if not isinstance(x, (jax.Array, np.ndarray)):
    x = jnp.asarray(x)

  if jnp.issubdtype(x.dtype, jnp.floating):
    return x
  elif jnp.issubdtype(x.dtype, jnp.integer):
    return x.astype(jnp.float_)
  else:
    raise ValueError(
        f"Expected either floating or integer dtype, got {x.dtype}.")
