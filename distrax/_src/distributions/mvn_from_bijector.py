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
"""MultivariateNormalFromBijector distribution."""

from typing import Callable, Union

import chex

from distrax._src.bijectors import block
from distrax._src.bijectors import chain
from distrax._src.bijectors import diag_linear
from distrax._src.bijectors import linear
from distrax._src.bijectors import shift
from distrax._src.distributions import independent
from distrax._src.distributions import normal
from distrax._src.distributions import transformed

import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array


def _check_input_parameters_are_valid(scale: linear.Linear, loc: Array) -> None:
  """Raises an error if `scale` and `loc` are not valid."""
  if loc.ndim < 1:
    raise ValueError('`loc` must have at least 1 dimension.')
  if scale.event_dims != loc.shape[-1]:
    raise ValueError(
        f'`scale` and `loc` have inconsistent dimensionality: '
        f'`scale.event_dims = {scale.event_dims} and '
        f'`loc.shape[-1] = {loc.shape[-1]}.')


class MultivariateNormalFromBijector(transformed.Transformed):
  """Multivariate normal distribution on `R^k`.

  The multivariate normal over `x` is characterized by an invertible affine
  transformation `x = f(z) = A @ z + b`, where `z` is a random variable that
  follows a standard multivariate normal on `R^k`, i.e., `p(z) = N(0, I_k)`,
  `A` is a `k x k` transformation matrix, and `b` is a `k`-dimensional vector.

  The resulting PDF on `x` is a multivariate normal, `p(x) = N(b, C)`, where
  `C = A @ A.T` is the covariance matrix. Additional leading dimensions (if any)
  index batches.

  The transformation `x = f(z)` must be specified by a linear scale bijector
  implementing the operation `A @ z` and a shift (or location) term `b`.
  """

  def __init__(self, loc: Array, scale: linear.Linear):
    """Initializes the distribution.

    Args:
      loc: The term `b`, i.e., the mean of the multivariate normal distribution.
      scale: The bijector specifying the linear transformation `A @ z`, as
        described in the class docstring.
    """
    _check_input_parameters_are_valid(scale, loc)
    batch_shape = jnp.broadcast_shapes(scale.batch_shape, loc.shape[:-1])
    dtype = jnp.result_type(scale.dtype, loc.dtype)

    # Build a standard multivariate Gaussian with the right `batch_shape`.
    std_mvn_dist = independent.Independent(
        distribution=normal.Normal(
            loc=jnp.zeros(batch_shape + loc.shape[-1:], dtype=dtype),
            scale=1.),
        reinterpreted_batch_ndims=1)
    # Form the bijector `f(x) = Ax + b`.
    bijector = chain.Chain([block.Block(shift.Shift(loc), ndims=1), scale])
    super().__init__(distribution=std_mvn_dist, bijector=bijector)
    self._scale = scale
    self._loc = loc
    self._event_shape = loc.shape[-1:]
    self._batch_shape = batch_shape
    self._dtype = dtype

  @property
  def scale(self) -> linear.Linear:
    """The scale bijector."""
    return self._scale

  @property
  def loc(self) -> Array:
    """The `loc` parameter of the distribution."""
    shape = self.batch_shape + self.event_shape
    return jnp.broadcast_to(self._loc, shape=shape)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def median(self) -> Array:
    """Calculates the median."""
    return self.loc

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.loc

  def covariance(self) -> Array:
    """Calculates the covariance matrix.

    Unlike TFP, which would drop leading dimensions, in Distrax the covariance
    matrix always has shape `batch_shape + (num_dims, num_dims)`. This helps to
    keep things simple and predictable.

    Returns:
      The covariance matrix, of shape `k x k` (broadcasted to match the batch
      shape of the distribution).
    """
    if isinstance(self.scale, diag_linear.DiagLinear):
      result = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(self.variance())
    else:
      result = jax.vmap(self.scale.forward, in_axes=-2, out_axes=-2)(
          self._scale.matrix)
    return jnp.broadcast_to(
        result, self.batch_shape + self.event_shape + self.event_shape)

  def variance(self) -> Array:
    """Calculates the variance of all one-dimensional marginals."""
    if isinstance(self.scale, diag_linear.DiagLinear):
      result = jnp.square(self.scale.diag)
    else:
      scale_matrix = self._scale.matrix
      result = jnp.sum(scale_matrix * scale_matrix, axis=-1)
    return jnp.broadcast_to(result, self.batch_shape + self.event_shape)

  def stddev(self) -> Array:
    """Calculates the standard deviation (the square root of the variance)."""
    if isinstance(self.scale, diag_linear.DiagLinear):
      result = jnp.abs(self.scale.diag)
    else:
      result = jnp.sqrt(self.variance())
    return jnp.broadcast_to(result, self.batch_shape + self.event_shape)


MultivariateNormalLike = Union[
    MultivariateNormalFromBijector, tfd.MultivariateNormalLinearOperator]


def _squared_frobenius_norm(x: Array) -> Array:
  """Computes the squared Frobenius norm of a matrix."""
  return jnp.sum(jnp.square(x), axis=[-2, -1])


def _log_abs_determinant(d: MultivariateNormalLike) -> Array:
  """Obtains `log|det(A)|`."""
  if isinstance(d, MultivariateNormalFromBijector):
    log_det_scale = d.scale.forward_log_det_jacobian(
        jnp.zeros(d.event_shape, dtype=d.dtype))
  elif isinstance(d, tfd.MultivariateNormalLinearOperator):
    log_det_scale = d.scale.log_abs_determinant()
  return log_det_scale


def _inv_scale_operator(d: MultivariateNormalLike) -> Callable[[Array], Array]:
  """Gets the operator that performs `A^-1 * x`."""
  if isinstance(d, MultivariateNormalFromBijector):
    inverse_fn = jax.vmap(d.scale.inverse, in_axes=-1, out_axes=-1)
  elif isinstance(d, tfd.MultivariateNormalLinearOperator):
    inverse_fn = d.scale.solve
  return inverse_fn


def _scale_matrix(d: MultivariateNormalLike) -> Array:
  """Gets the full scale matrix `A`."""
  if isinstance(d, MultivariateNormalFromBijector):
    matrix = d.scale.matrix
  elif isinstance(d, tfd.MultivariateNormalLinearOperator):
    matrix = d.scale.to_dense()
  return matrix


def _has_diagonal_scale(d: MultivariateNormalLike) -> bool:
  """Determines if the scale matrix `A` is diagonal."""
  if (isinstance(d, MultivariateNormalFromBijector)
      and isinstance(d.scale, diag_linear.DiagLinear)):
    return True
  elif (isinstance(d, tfd.MultivariateNormalDiag) or
        (isinstance(d, tfd.MultivariateNormalFullCovariance) and
         d.parameters['covariance_matrix'] is None) or
        (isinstance(d, tfd.MultivariateNormalTriL) and
         not isinstance(d, tfd.MultivariateNormalFullCovariance) and
         d.parameters['scale_tril'] is None) or
        (isinstance(d, tfd.MultivariateNormalDiagPlusLowRank) and
         d.parameters['scale_perturb_factor'] is None)):
    return True
  return False


def _kl_divergence_mvn_mvn(
    dist1: MultivariateNormalLike,
    dist2: MultivariateNormalLike,
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Divergence KL(dist1 || dist2) between multivariate normal distributions.

  Args:
    dist1: A multivariate normal distribution.
    dist2: A multivariate normal distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """

  num_dims = tuple(dist1.event_shape)[-1]  # `tuple` needed for TFP distrib.
  if num_dims != tuple(dist2.event_shape)[-1]:
    raise ValueError(f'Both multivariate normal distributions must have the '
                     f'same `event_shape`, but they have {num_dims} and '
                     f'{tuple(dist2.event_shape)[-1]} dimensions.')

  # Calculation is based on:
  # https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/distributions/mvn_linear_operator.py#L384
  # If C_1 = AA.T, C_2 = BB.T, then
  #   tr[inv(C_2) C_1] = ||inv(B) A||_F^2
  # where ||.||_F^2 is the squared Frobenius norm.
  diff_lob_abs_det = _log_abs_determinant(dist2) - _log_abs_determinant(dist1)
  if _has_diagonal_scale(dist1) and _has_diagonal_scale(dist2):
    # This avoids instantiating the full scale matrix when it is diagonal.
    b_inv_a = jnp.expand_dims(dist1.stddev() / dist2.stddev(), axis=-1)
  else:
    b_inv_a = _inv_scale_operator(dist2)(_scale_matrix(dist1))
  diff_mean_expanded = jnp.expand_dims(dist2.mean() - dist1.mean(), axis=-1)
  b_inv_diff_mean = _inv_scale_operator(dist2)(diff_mean_expanded)
  kl_divergence = (
      diff_lob_abs_det +
      0.5 * (-num_dims +
             _squared_frobenius_norm(b_inv_a) +
             _squared_frobenius_norm(b_inv_diff_mean)))
  return kl_divergence


# Register the KL functions with TFP.
tfd.RegisterKL(
    MultivariateNormalFromBijector, MultivariateNormalFromBijector)(
        _kl_divergence_mvn_mvn)
tfd.RegisterKL(
    MultivariateNormalFromBijector, tfd.MultivariateNormalLinearOperator)(
        _kl_divergence_mvn_mvn)
tfd.RegisterKL(
    tfd.MultivariateNormalLinearOperator, MultivariateNormalFromBijector)(
        _kl_divergence_mvn_mvn)
