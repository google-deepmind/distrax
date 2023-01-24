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
"""Utility functions for performing Bijector-related JAX transformations.

A typical bijector will implement the following functionality:

- `forward`: the transformation of samples from an underlying distribution.
- `inverse`: the inverse function of `forward`.
- `[forward|inverse]_log_det_jacobian`: computes the log of the absolute value
    of the determinant of the Jacobian of the bijector's forward or inverse
    functions at a particular point.
- `is_constant_jacobian`: a boolean indicating whether the Jacobian of the
    forward and inverse functions is constant with respect to the function's
    input.

This module provides the following utilities for deriving these functions
automatically from JAX code implementing either the `forward` or `inverse`
functions:

- `inv`: inverts an input function `f` as long as it is composed of invertible
    primitives. This is achieved by tracing `f` to its underlying JAXPR
    representation and interpreting the code in reverse, substituting
    each invertible primitive by its inverse. Once transformed by `jit`, the
    code produced by the inversion interpreter is unrolled and JAX sees only the
    resulting inverse operations.
    The module provides the `register_inverse` function to specify custom
    inversion rules for primitives not yet supported here.
    See official JAX documentation for more information on tracing,
    writing custom interpreters, and a simple example of function inversion.
- `log_det_scalar`: computes the log-determinant of the Jacobian of a scalar
    function, using JAX autograd machinery.
- `is_constant_jacobian`: attempts to determine whether the Jacobian of `f` is
    constant with respect to its input by tracing `f` to its underlying JAXPR
    representation and checking whether any transformations of the input
    appear in the output. This is an experimental feature.
"""

import functools

from absl import logging
import jax
import jax.numpy as jnp


_inverse_registry = {
    # unary ops
    jax.lax.tanh_p: jax.lax.atanh_p,
    jax.lax.atanh_p: jax.lax.tanh_p,
    jax.lax.sinh_p: jax.lax.asinh_p,
    jax.lax.asinh_p: jax.lax.sinh_p,
    jax.lax.cosh_p: jax.lax.acosh_p,
    jax.lax.acosh_p: jax.lax.cosh_p,
    jax.lax.exp_p: jax.lax.log_p,
    jax.lax.log_p: jax.lax.exp_p,
    jax.lax.sqrt_p: lambda x: jax.lax.pow_p.bind(x, 2.0),
    jax.lax.rsqrt_p: lambda x: 1.0 / jax.lax.pow_p.bind(x, 2.0),
    jax.lax.neg_p: jax.lax.neg_p,
    jax.lax.log1p_p: jax.lax.expm1_p,
    jax.lax.expm1_p: jax.lax.log1p_p,
    jax.lax.erf_p: jax.lax.erf_inv_p,
    jax.lax.erf_inv_p: jax.lax.erf_p,
    jax.lax.conj_p: jax.lax.conj_p,

    # binary ops; tuple values represent the variable-left/variable-right side
    # case for non-commutatively invertible ops like div
    jax.lax.mul_p: (jax.lax.div_p.bind, lambda x, y: jax.lax.div_p.bind(y, x)),
    jax.lax.div_p: (jax.lax.mul_p.bind, jax.lax.div_p.bind),
    jax.lax.add_p: (jax.lax.sub_p.bind, lambda x, y: jax.lax.sub_p.bind(y, x)),
    jax.lax.sub_p: (jax.lax.add_p.bind, jax.lax.sub_p.bind),
    jax.lax.pow_p: lambda x, y: jax.lax.pow_p.bind(x, 1.0/y),
    jax.lax.integer_pow_p: lambda x, y: jax.lax.pow_p.bind(x, 1.0/y)
}


_potentially_unstable_primitives = {
    jax.lax.tanh_p: "distrax.Tanh or distrax.Inverse(distrax.Tanh)",
    jax.lax.atanh_p: "distrax.Tanh or distrax.Inverse(distrax.Tanh)",
}


def register_inverse(primitive, inverse_left, inverse_right=None):
  """Register a function that implements the inverse of a JAX primitive.

  Args:
    primitive: JAX primitive, often named `*_p` and located in `jax.lax.lax.py`.
    inverse_left: a function implementing the inverse if the primitive is
        a unary operator or if `inv(f(x,y)) == inv(f(y,x))`, else a function
        implementing the inverse of a binary operator when the variable in
        question comes before the operator, e.g. `x div_p 2`.
    inverse_right: a function implementing the inverse of a binary
        operator when the variable in question comes after the operator,
        e.g. `2 div_p x`.
  """
  if inverse_right is None:
    _inverse_registry[primitive] = inverse_left
  else:
    _inverse_registry[primitive] = (inverse_left, inverse_right)


def inv(fun):
  """Returns the inverse of `fun` such that (inv(fun) o fun)(x) = x."""
  jaxpr_fn = _invertible_jaxpr_and_constants(fun)

  @functools.wraps(fun)  # pylint: disable=no-value-for-parameter
  def wrapped(*args, **kwargs):
    jaxpr, consts = jaxpr_fn(*args, **kwargs)
    out = _interpret_inverse(jaxpr, consts, *args)
    return out[0]
  return wrapped


def is_constant_jacobian(fn, x=0.0):
  """Experimental. Attempts to determine whether `fn` has a constant Jacobian.

  This function attempts to determine whether the Jacobian of `fn` is constant
  w.r.t. its input. We compute the Jacobian of `fn` at `x` and inspect the
  jaxpr to determine whether any function of the input appears at the output.

  Args:
    fn: a JAX-traceable differentiable function taking scalar input.
    x: the location at which to check whether the Jacobian is constant.

  Returns:
    Boolean value indicating whether the Jacobian is constant at `x`.
  """

  jac_fn = jax.jacfwd(fn)
  jac_jaxpr = jax.make_jaxpr(jac_fn)(jnp.array(x)).jaxpr
  dependent_vars = _dependent_variables(jac_jaxpr)

  jac_is_constant = not any(isinstance(v, jax.core.Var) and v in dependent_vars
                            for v in jac_jaxpr.outvars)

  return jac_is_constant


def log_det_scalar(fn):
  """Uses JAX autograd to derive the log-det-jacobian of a scalar function."""
  _check_numerical_stability(fn)
  jac_fn = jax.vmap(jax.jacfwd(fn))
  def log_det_fn(x):
    x = jnp.asarray(x)
    jac_scalar = jac_fn(x.reshape(-1))
    log_det_ = jnp.log(jnp.absolute(jac_scalar))
    return log_det_.reshape(x.shape)
  return log_det_fn


def _check_numerical_stability(fn):
  """Logs a warning if numerically unstable operations are requested."""
  jaxpr = jax.make_jaxpr(fn)(0.0).jaxpr
  for eqn in jaxpr.eqns:
    if eqn.primitive in _potentially_unstable_primitives:
      logging.warn("[Distrax]: the '%s' primitive can exhibit unstable "
                   "numerical behavior under certain circumstances. Consider "
                   "using the %s bijector instead if possible.", eqn.primitive,
                   _potentially_unstable_primitives[eqn.primitive])


def _dependent_variables(jaxpr, dependent=None):
  """Returns the set of variables in the jaxpr that depend on the input vars."""
  if dependent is None:
    dependent = set(jaxpr.invars)

  for eqn in jaxpr.eqns:
    # If primitive is an xla_call, get subexpressions and evaluate recursively
    call_jaxpr, _ = _extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      to_name = dict(zip(eqn.invars, call_jaxpr.invars))
      arg_dependence = set(to_name[v] for v in eqn.invars if v in dependent)
      subjaxpr_dependent = _dependent_variables(call_jaxpr, arg_dependence)
      from_name = dict(zip(call_jaxpr.outvars, eqn.outvars))
      dependent.update(from_name[v] for v in call_jaxpr.outvars
                       if v in subjaxpr_dependent)
    else:
      for v in eqn.invars:
        if isinstance(v, jax.core.Var) and v in dependent:
          dependent.update(eqn.outvars)

  return dependent


def _invertible_jaxpr_and_constants(fun):
  """Returns a transformation from function invocation to invertible jaxpr."""
  jaxpr_maker = jax.make_jaxpr(fun)

  @functools.wraps(fun)  # pylint: disable=no-value-for-parameter
  def jaxpr_const_maker(*args, **kwargs):
    typed_jaxpr = jaxpr_maker(*args, **kwargs)
    return typed_jaxpr.jaxpr, typed_jaxpr.literals
  return jaxpr_const_maker


def _identify_variable_in_eqn(eqn):
  """Identify whether primitive is a unop or binop and which side var is on."""

  if len(eqn.invars) == 1:  # unary operation
    var_idx = 0

  elif len(eqn.invars) == 2:  # binary operation
    if tuple(map(type, eqn.invars)) == (jax.core.Var, jax.core.Literal):
      var_idx = 0

    elif tuple(map(type, eqn.invars)) == (jax.core.Literal, jax.core.Var):
      var_idx = 1

    elif tuple(map(type, eqn.invars)) == (jax.core.Var, jax.core.Var):
      raise NotImplementedError(
          "Expressions with multiple occurrences of the input variable are "
          "not supported. Please rearrange such that the variable appears only "
          "once in the expression if possible. If not possible, consider "
          "providing both `forward` and `inverse` to Lambda explicitly.")

    elif tuple(map(type, eqn.invars)) == (jax.core.Literal, jax.core.Literal):
      raise ValueError("Expression appears to contain no variables and "
                       "therefore cannot be inverted.")

    else:
      raise NotImplementedError("Unsupported binary op combination: "
                                + str(tuple(map(type, eqn.invars))))

  else:
    raise NotImplementedError(f"Op {eqn.primitive} with cardinality >= 3 not "
                              "supported.")

  return var_idx


def _interpret_inverse(jaxpr, consts, *args):
  """Interprets and executes the inverse of `jaxpr`."""
  env = {}

  def read(var):
    return var.val if isinstance(var, jax.core.Literal) else env[var]
  def write(var, val):
    env[var] = val

  jax.api_util.safe_map(write, jaxpr.outvars, args)
  jax.api_util.safe_map(write, jaxpr.constvars, consts)

  for eqn in reversed(jaxpr.eqns):
    params = eqn.params.copy()
    # identify the cardinality of the op and the index of the variable in eqn
    var_idx = _identify_variable_in_eqn(eqn)

    # if primitive is an xla_call, get subexpressions and evaluate recursively
    call_jaxpr, params = _extract_call_jaxpr(eqn.primitive, params)
    if call_jaxpr:
      subfuns = [jax.linear_util.wrap_init(
          functools.partial(_interpret_inverse, call_jaxpr, ()))]
      prim_inv = eqn.primitive

    elif eqn.primitive is jax.experimental.pjit.pjit_p:
      pjit_jaxpr = params.pop("jaxpr")
      partial_inverse = functools.partial(_interpret_inverse, pjit_jaxpr.jaxpr,
                                          pjit_jaxpr.consts)
      inverse_jaxpr = jax.make_jaxpr(partial_inverse)(*args)
      params["jaxpr"] = inverse_jaxpr

      prim_inv = eqn.primitive
      subfuns = []

    else:  # otherwise, get its inverse if it exists
      if eqn.primitive not in _inverse_registry:
        raise NotImplementedError(
            f"Primitive '{eqn.primitive}' does not have a registered inverse.")

      # use the correct inverse formulation depending on whether the variable is
      # on the left or right side of the expression
      prim_inv = _inverse_registry[eqn.primitive]
      if isinstance(prim_inv, tuple):
        prim_inv = prim_inv[var_idx]

      subfuns = []

    # get the values of any variables in the eqn
    invals = jax.api_util.safe_map(read, eqn.outvars)

    # place the args and variables in the right order
    if var_idx == 0:
      prim_args = subfuns + invals + [v.val for v in eqn.invars[1:]]
    else:
      prim_args = subfuns + [v.val for v in eqn.invars[:1]] + invals

    # if the inverse is a primitive, bind it, otherwise call it directly
    if hasattr(prim_inv, "bind"):
      outvals = prim_inv.bind(*prim_args, **params)
    else:
      outvals = prim_inv(*prim_args, **params)

    # if the primitive returns multiple results, write them all to env
    if (hasattr(prim_inv, "multiple_results") and prim_inv.multiple_results):
      jax.api_util.safe_map(write, eqn.invars, outvals)
    else:
      write(eqn.invars[var_idx], outvals)

  if any(v not in env for v in jaxpr.invars):
    raise ValueError("Expression appears to contain no variables and therefore "
                     "cannot be inverted.")

  return jax.api_util.safe_map(read, jaxpr.invars)


def _extract_call_jaxpr(primitive, params):
  if not (primitive.call_primitive or primitive.map_primitive):
    return None, params
  else:
    params = dict(params)
    return params.pop("call_jaxpr"), params
