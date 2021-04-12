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
"""Utility math functions."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

Array = chex.Array


def multiply_no_nan(x: Array, y: Array) -> Array:
  """Equivalent of TF `multiply_no_nan`.

  Computes the element-wise product of `x` and `y` and return 0 if `y` is zero,
  even if `x` is NaN or infinite.

  Args:
    x: First input.
    y: Second input.

  Returns:
    The product of `x` and `y`.

  Raises:
    ValueError if the shapes of `x` and `y` do not match.
  """
  dtype = jnp.result_type(x, y)
  return jnp.where(y == 0, jnp.zeros((), dtype=dtype), x * y)


def power_no_nan(x: Array, y: Array) -> Array:
  """Computes `x ** y` and ensure that the result is 1.0 when `y` is zero.

  Compute the element-wise power `x ** y` and return 1.0 when `y` is zero,
  regardless of the value of `x`, even if it is NaN or infinite. This method
  uses the convention `0 ** 0 = 1`.

  Args:
    x: First input.
    y: Second input.

  Returns:
    The power `x ** y`.
  """
  dtype = jnp.result_type(x, y)
  return jnp.where(y == 0, jnp.ones((), dtype=dtype), jnp.power(x, y))


def normalize(
    *, probs: Optional[Array] = None, logits: Optional[Array] = None) -> Array:
  """Normalize logits (via log_softmax) or probs (ensuring they sum to one)."""
  if logits is None:
    probs = jnp.asarray(probs)
    return probs / probs.sum(axis=-1, keepdims=True)
  else:
    logits = jnp.asarray(logits)
    return jax.nn.log_softmax(logits, axis=-1)


def sum_last(x: Array, ndims: int) -> Array:
  """Sums the last `ndims` axes of array `x`."""
  axes_to_sum = tuple(range(-ndims, 0))
  return jnp.sum(x, axis=axes_to_sum)


def log_expbig_minus_expsmall(big: Array, small: Array) -> Array:
  """Stable implementation of `log(exp(big) - exp(small))`.

  Args:
    big: First input.
    small: Second input. It must be `small <= big`.

  Returns:
    The resulting `log(exp(big) - exp(small))`.
  """
  return big + jnp.log1p(-jnp.exp(small - big))
