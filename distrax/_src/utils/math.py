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

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

Array = chex.Array


@jax.custom_jvp
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


@multiply_no_nan.defjvp
def multiply_no_nan_jvp(
    primals: Tuple[Array, Array],
    tangents: Tuple[Array, Array]) -> Tuple[Array, Array]:
  """Custom gradient computation for `multiply_no_nan`."""
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = multiply_no_nan(x, y)
  tangent_out = y * x_dot + x * y_dot
  return primal_out, tangent_out


@jax.custom_jvp
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


@power_no_nan.defjvp
def power_no_nan_jvp(
    primals: Tuple[Array, Array],
    tangents: Tuple[Array, Array]) -> Tuple[Array, Array]:
  """Custom gradient computation for `power_no_nan`."""
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = power_no_nan(x, y)
  tangent_out = (y * power_no_nan(x, y - 1) * x_dot
                 + primal_out * jnp.log(x) * y_dot)
  return primal_out, tangent_out


def mul_exp(x: Array, logp: Array) -> Array:
  """Returns `x * exp(logp)` with zero output if `exp(logp)==0`.

  Args:
    x: An array.
    logp: An array.

  Returns:
    `x * exp(logp)` with zero output and zero gradient if `exp(logp)==0`,
    even if `x` is NaN or infinite.
  """
  p = jnp.exp(logp)
  # If p==0, the gradient with respect to logp is zero,
  # so we can replace the possibly non-finite `x` with zero.
  x = jnp.where(p == 0, 0.0, x)
  return x * p


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


def log_beta(a: Array, b: Array) -> Array:
  """Obtains the log of the beta function `log B(a, b)`.

  Args:
    a: First input. It must be positive.
    b: Second input. It must be positive.

  Returns:
    The value `log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a + b)`,
    where `Gamma` is the Gamma function, obtained through stable computation of
    `log Gamma`.
  """
  return jax.lax.lgamma(a) + jax.lax.lgamma(b) - jax.lax.lgamma(a + b)


def log_beta_multivariate(a: Array) -> Array:
  """Obtains the log of the multivariate beta function `log B(a)`.

  Args:
    a: An array of length `K` containing positive values.

  Returns:
    The value
    `log B(a) = sum_{k=1}^{K} log Gamma(a_k) - log Gamma(sum_{k=1}^{K} a_k)`,
    where `Gamma` is the Gamma function, obtained through stable computation of
    `log Gamma`.
  """
  return (
      jnp.sum(jax.lax.lgamma(a), axis=-1) - jax.lax.lgamma(jnp.sum(a, axis=-1)))
