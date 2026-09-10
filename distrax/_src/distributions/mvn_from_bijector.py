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
import jax.scipy as jsp

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

  def symmetrized_kl_divergence(self, other: 'MultivariateNormalLike') -> Array:
    """Computes KL(self || other) + KL(other || self) without subtractive cancellation."""
    m, _ = _cholesky_cross_operator_from_dist(self, other)
    cov_term = 0.5 * jnp.sum(jnp.square(m), axis=(-2, -1))

    # pyrefly: ignore[bad-index,unsupported-operation]
    delta_mu = (self.mean() - other.mean())[..., None]
    v1 = _inv_scale_operator(self)(delta_mu)
    v2 = _inv_scale_operator(other)(delta_mu)
    mean_term = 0.5 * jnp.sum(jnp.square(v1) + jnp.square(v2), axis=(-2, -1))

    return cov_term + mean_term

  def bhattacharyya_distance(self, other: 'MultivariateNormalLike') -> Array:
    """Computes the Bhattacharyya distance without subtractive cancellation."""
    _, k = _cholesky_cross_operator_from_dist(self, other)
    cov_term = 0.25 * _logdet_one_plus_k(k)

    # pyrefly: ignore[bad-index,unsupported-operation]
    delta_mu = (self.mean() - other.mean())[..., None]
    sigma_avg = 0.5 * (self.covariance() + other.covariance())
    l_avg = jnp.linalg.cholesky(sigma_avg)
    v_avg = jsp.linalg.solve_triangular(l_avg, delta_mu, lower=True)
    mean_term = 0.125 * jnp.sum(jnp.square(v_avg), axis=(-2, -1))

    return cov_term + mean_term

  def geometric_jensen_shannon_divergence(
      self, other: 'MultivariateNormalLike'
  ) -> Array:
    """Computes the Geometric Jensen-Shannon Divergence without subtractive cancellation."""
    _, k = _cholesky_cross_operator_from_dist(self, other)
    cov_term = 0.25 * _trace_two_k_minus_log_one_plus_k(k)

    # pyrefly: ignore[bad-index,unsupported-operation]
    delta_mu = (self.mean() - other.mean())[..., None]
    v1 = _inv_scale_operator(self)(delta_mu)
    v2 = _inv_scale_operator(other)(delta_mu)

    sigma_avg = 0.5 * (self.covariance() + other.covariance())
    l_avg = jnp.linalg.cholesky(sigma_avg)
    v_avg = jsp.linalg.solve_triangular(l_avg, delta_mu, lower=True)

    mean_term = 0.125 * jnp.sum(
        jnp.square(v1) + jnp.square(v2), axis=(-2, -1)
    ) - 0.125 * jnp.sum(jnp.square(v_avg), axis=(-2, -1))
    mean_term = jnp.maximum(mean_term, 0.0)

    return cov_term + mean_term


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
  else:
    raise ValueError(f'Unsupported distribution: {type(d)}.')
  return log_det_scale


def _inv_scale_operator(d: MultivariateNormalLike) -> Callable[[Array], Array]:
  """Gets the operator that performs `A^-1 * x`."""
  if isinstance(d, MultivariateNormalFromBijector):
    inverse_fn = jax.vmap(d.scale.inverse, in_axes=-1, out_axes=-1)
  elif isinstance(d, tfd.MultivariateNormalLinearOperator):
    inverse_fn = d.scale.solve
  else:
    raise ValueError(f'Unsupported distribution: {type(d)}.')
  return inverse_fn


def _scale_matrix(d: MultivariateNormalLike) -> Array:
  """Gets the full scale matrix `A`."""
  if isinstance(d, MultivariateNormalFromBijector):
    matrix = d.scale.matrix
  elif isinstance(d, tfd.MultivariateNormalLinearOperator):
    matrix = d.scale.to_dense()
  else:
    raise ValueError(f'Unsupported distribution: {type(d)}.')
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


def _cholesky_cross_operator_from_dist(dist1, dist2):
  """Computes cross-operator M and normalized Gramian K directly from bijectors."""
  x = _inv_scale_operator(dist2)(_scale_matrix(dist1))
  y = _inv_scale_operator(dist1)(_scale_matrix(dist2))
  m = x - jnp.swapaxes(y, -1, -2)
  k = 0.25 * jnp.matmul(m, jnp.swapaxes(m, -1, -2))
  # Ensure exact symmetry
  k = 0.5 * (k + jnp.swapaxes(k, -1, -2))
  return m, k


def _derive_taylor_threshold(dtype: jnp.dtype, max_order: int = 4) -> float:
  eps = float(jnp.finfo(dtype).eps)
  return float((eps * (max_order + 1)) ** (1.0 / (max_order + 1)))


def _logdet_one_plus_k(k: jax.Array) -> jax.Array:
  """Computes ln det(I + K) for PSD K, stable for small ||K||."""
  tr_k2 = jnp.sum(jnp.square(k), axis=(-2, -1))
  norm_k = jnp.sqrt(tr_k2)
  threshold = _derive_taylor_threshold(k.dtype)

  # Since K and K^2 are symmetric, tr(K^3) = <K, K^2>_F and
  # tr(K^4) = ||K^2||_F^2, requiring only a single matrix multiplication
  # (K^2 = K @ K).
  k2 = jnp.matmul(k, k)
  tr_k = jnp.trace(k, axis1=-2, axis2=-1)
  tr_k3 = jnp.sum(k * k2, axis=(-2, -1))
  tr_k4 = jnp.sum(jnp.square(k2), axis=(-2, -1))
  logdet_taylor = (
      tr_k - 0.5 * tr_k2 + (1.0 / 3.0) * tr_k3 - 0.25 * tr_k4
  )

  # Large-norm Cholesky: 2 * sum(log(diag(chol(I + K))))
  eye = jnp.eye(k.shape[-1], dtype=k.dtype)
  l_k = jnp.linalg.cholesky(eye + k)
  logdet_chol = 2.0 * jnp.sum(
      jnp.log(jnp.diagonal(l_k, axis1=-2, axis2=-1)), axis=-1
  )
  return jnp.where(norm_k < threshold, logdet_taylor, logdet_chol)


def _trace_two_k_minus_log_one_plus_k(k: jax.Array) -> jax.Array:
  """Computes tr(2K - ln(I + K)) for PSD K without subtractive cancellation."""
  tr_k2 = jnp.sum(jnp.square(k), axis=(-2, -1))
  norm_k = jnp.sqrt(tr_k2)
  threshold = _derive_taylor_threshold(k.dtype)
  tr_k = jnp.trace(k, axis1=-2, axis2=-1)

  # Small-norm Taylor expansion: tr(K + K^2/2 - K^3/3 + K^4/4)
  k2 = jnp.matmul(k, k)
  tr_k3 = jnp.sum(k * k2, axis=(-2, -1))
  tr_k4 = jnp.sum(jnp.square(k2), axis=(-2, -1))
  val_taylor = (
      tr_k + 0.5 * tr_k2 - (1.0 / 3.0) * tr_k3 + 0.25 * tr_k4
  )

  # Large-norm Cholesky: 2 tr(K) - ln det(I + K)
  eye = jnp.eye(k.shape[-1], dtype=k.dtype)
  l_k = jnp.linalg.cholesky(eye + k)
  logdet_chol = 2.0 * jnp.sum(
      jnp.log(jnp.diagonal(l_k, axis1=-2, axis2=-1)), axis=-1
  )
  val_chol = jnp.maximum(2.0 * tr_k - logdet_chol, 0.0)
  return jnp.where(norm_k < threshold, val_taylor, val_chol)


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
  # pyrefly: ignore[unsupported-operation]
  diff_lob_abs_det = _log_abs_determinant(dist2) - _log_abs_determinant(dist1)
  if _has_diagonal_scale(dist1) and _has_diagonal_scale(dist2):
    # This avoids instantiating the full scale matrix when it is diagonal.
    b_inv_a = jnp.expand_dims(dist1.stddev() / dist2.stddev(), axis=-1)
  else:
    b_inv_a = _inv_scale_operator(dist2)(_scale_matrix(dist1))
  # pyrefly: ignore[unsupported-operation]
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
