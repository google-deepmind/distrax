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

import functools
from typing import Optional, Tuple

import chex
import jax
from jax.custom_derivatives import SymbolicZero
import jax.numpy as jnp

Array = chex.Array


def _zero_where_y_is_zero(x: Array, y: Array) -> Array:
  """Replaces the entries of `x` where `y` is zero by zeros.

  Multiplying this by `y` evaluates `0 * y` rather than `x * 0`, so an infinite
  `x` does not produce an intermediate NaN. The mask only depends on primals,
  which keeps the JVP rule reusing it linear in the tangents, as reverse-mode
  transposition requires.

  Args:
    x: The value to mask.
    y: The value whose zeros select the entries to replace.

  Returns:
    `x` with zeros wherever `y` is zero.
  """
  return jnp.where(y == 0, jnp.zeros((), dtype=jnp.result_type(x, y)), x)


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
  # Replace `x` with zero where `y` is zero before multiplying, so that `0 * y`
  # is computed instead of `x * 0`. This avoids producing an intermediate NaN
  # when `x` is infinite and `y` is zero, which would be detected by
  # `checkify`'s NaN checks even though the result is correct.
  return _zero_where_y_is_zero(x, y) * y


# TODO(dougalm): move helpers like these into JAX AD utils
def add_maybe_symbolic(x, y):
  if isinstance(x, SymbolicZero):
    return y
  elif isinstance(y, SymbolicZero):
    return x
  else:
    return x + y


def scale_maybe_symbolic(result_aval, tangent, scale):
  if isinstance(tangent, SymbolicZero):
    return SymbolicZero(result_aval)
  else:
    return tangent * scale


def mask_tangent_maybe_symbolic(tangent, y):
  """Zeros a tangent where `y` is zero, leaving symbolic zeros untouched."""
  if isinstance(tangent, SymbolicZero):
    return tangent
  else:
    return _zero_where_y_is_zero(tangent, y)


@functools.partial(multiply_no_nan.defjvp, symbolic_zeros=True)
def multiply_no_nan_jvp(
    primals: Tuple[Array, Array],
    tangents: Tuple[Array, Array]) -> Tuple[Array, Array]:
  """Custom gradient computation for `multiply_no_nan`."""
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = multiply_no_nan(x, y)
  primal_aval = jax.typeof(primal_out)
  result_aval = primal_aval.at_least_vspace()
  # The derivative with respect to `y` is `x`, but it is taken to be zero where
  # `y` is zero, matching the primal. The tangents are masked the same way,
  # because an infinite tangent multiplied by the zero it contributes would
  # otherwise evaluate `inf * 0`, and `checkify`'s NaN checks report that even
  # though the term is discarded. Every mask depends on the primal `y` alone,
  # which keeps the rule linear in the tangents, as reverse-mode transposition
  # requires.
  tangent_out_1 = scale_maybe_symbolic(result_aval,
                                       mask_tangent_maybe_symbolic(x_dot, y), y)
  tangent_out_2 = scale_maybe_symbolic(
      result_aval, mask_tangent_maybe_symbolic(y_dot, y),
      _zero_where_y_is_zero(x, y))
  return primal_out, add_maybe_symbolic(tangent_out_1, tangent_out_2)


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
  # pyrefly: ignore[unsupported-operation]
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
