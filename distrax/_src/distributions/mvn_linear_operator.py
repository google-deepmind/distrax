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
"""MultivariateNormalLinearOperator distribution."""

from typing import Union

import chex
from distrax._src.distributions import independent
from distrax._src.distributions import normal
from distrax._src.distributions import transformed
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfb = tfp.bijectors
tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey


class MultivariateNormalLinearOperator(transformed.Transformed):
  """Multivariate normal distribution on `R^k`.

  The multivariate normal over `x` is characterized by a linear invertible
  transformation `x = f(z) = A @ z + b`, where `z` is a random variable that
  follows a standard multivariate normal on `R^k`, i.e., `p(z) = N(0, I_k)`,
  `A` is a `k x k` transformation matrix, and `b` is a `k`-dimensional vector.
  Additional leading dimensions (if any) index batches.

  The resulting PDF on `x` is a multivariate normal, `p(x) = N(b, C)`, where
  `C = A @ A'` is the covariance matrix.

  The transformation `x = f(z)` must be fully specified by a TFP `Affine`
  bijector. Note that this departs from the TFP convention, where the
  transformation is specified by two parameters instead: `scale` (of class
  `LinearOperator`) and `loc` (an array).
  """

  equiv_tfp_cls = tfd.MultivariateNormalLinearOperator

  def __init__(self,
               bijector: tfb.Affine,
               dtype: jnp.dtype = jnp.float32):
    """Initializes the distribution.

    Args:
      bijector: The bijector specifying the linear transformation, as described
        in the class docstring.
      dtype: The type of the distribution samples.
    """
    if bijector.forward_min_event_ndims != 1:
      raise ValueError(f'Incorrect value for the `forward_min_event_ndims` '
                       f'attribute of the bijector: '
                       f'{bijector.forward_min_event_ndims} != 1.')
    if not bijector.is_constant_jacobian:
      raise ValueError('The bijector must have constant Jacobian.')
    if isinstance(bijector.scale, Array):
      raise ValueError('The bijector `scale` cannot be an `Array`. This '
                       'happens when the bijector was created by specifying '
                       'only `shift` and/or `scale_identity_multiplier`.')
    if len(bijector.scale.shape) < 2:
      raise ValueError(f'The bijector `scale` must have at least 2 dimensions. '
                       f'Got shape {bijector.scale.shape}.')
    if bijector.shift is None:
      raise ValueError('The bijector `shift` parameter cannot be `None`.')
    if not bijector.shift.shape:
      raise ValueError('The bijector `shift` must have at least 1 dimension.')

    # Build a standard multivariate Gaussian with the appropriate `num_dims`.
    num_dims = bijector.scale.shape[-1]
    std_mvn_dist = independent.Independent(
        distribution=normal.Normal(
            loc=jnp.zeros((num_dims,), dtype=dtype),
            scale=1., dtype=dtype),
        reinterpreted_batch_ndims=1)
    super().__init__(distribution=std_mvn_dist, bijector=bijector)

    self._num_dims = num_dims
    self._event_shape = (num_dims,)
    self._batch_shape = jax.lax.broadcast_shapes(
        bijector.shift.shape[:-1], tuple(bijector.scale.shape)[:-2])
    self._dtype = dtype

  @property
  def scale_operator(self):
    """The linear operator that corresponds to the bijector's `scale`."""
    return self.bijector.scale

  @property
  def _shift(self) -> Array:
    return jnp.asarray(self.bijector.shift, dtype=self.dtype)

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._shift, self.batch_shape + self.event_shape)

  @property
  def has_diagonal_scale(self) -> bool:
    """Whether the transformation is parameterized by a diagonal matrix."""
    return has_diagonal_scale(self.bijector)

  @property
  def num_dims(self) -> int:
    """Dimensionality of the events."""
    return self._num_dims

  @property
  def scale(self) -> Array:
    """Scale matrix defining the transformation."""
    return jnp.broadcast_to(
        jnp.asarray(self.scale_operator.to_dense(), dtype=self.dtype),
        self.batch_shape + self.event_shape + self.event_shape)

  @property
  def scale_diag_part(self) -> Array:
    """Diagonal part of the scale matrix that defines the transformation."""
    return jnp.broadcast_to(
        jnp.asarray(self.scale_operator.diag_part(), dtype=self.dtype),
        self.batch_shape + self.event_shape)

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def median(self) -> Array:
    """Calculates the marginal median."""
    return self.mean()

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    if self.has_diagonal_scale:
      std = jnp.abs(self.scale_diag_part)
    else:
      std = jnp.sqrt(
          jnp.vectorize(jnp.diag, signature='(k,k)->(k)')(self.covariance()))
    return std

  def variance(self) -> Array:
    """Calculates the variance."""
    if self.has_diagonal_scale:
      var = jnp.square(self.scale_diag_part)
    else:
      var = jnp.vectorize(jnp.diag, signature='(k,k)->(k)')(self.covariance())
    return var

  def covariance(self) -> Array:
    """Calculates the covariance matrix in `R^k x R^k`.

    Additional dimensions, if any, correspond to batch dimensions. Note that
    TFP drops the leading batch dimensions if the shift parameter `b` of the
    transformation has more batch dimensions than the matrix `scale`. To keep
    things simple and predictable, and for consistency with other methods, in
    Distrax the `covariance` has shape `batch_shape + (num_dims, num_dims)`.

    Returns:
      The covariance matrix of shape `batch_shape + (num_dims, num_dims)`.
    """
    if self.has_diagonal_scale:
      cov = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(
          jnp.square(self.scale_diag_part))
    else:
      cov = jnp.broadcast_to(
          jnp.asarray(self.scale_operator.matmul(self.scale_operator.to_dense(),
                                                 adjoint_arg=True),
                      dtype=self.dtype),
          self.batch_shape + self.event_shape + self.event_shape)
    return cov


def has_diagonal_scale(b: tfb.Affine) -> bool:
  """Determines whether a TFP `Afine` bijector has diagonal scale."""
  return (b.parameters['scale_tril'] is None and
          b.parameters['scale_perturb_factor'] is None)


MultivariateNormalLinearOperatorLike = Union[
    MultivariateNormalLinearOperator,
    tfd.MultivariateNormalLinearOperator]


def _kl_divergence_mvn_linear_op_mvn_linear_op(
    dist1: MultivariateNormalLinearOperatorLike,
    dist2: MultivariateNormalLinearOperatorLike,
    *unused_args, **unused_kwargs,
    ) -> Array:
  """Divergence KL(dist1 || dist2) between two MultivariateNormalLinearOperator.

  This implementation requires to compute `inv(C_2) C_1`, where `C_1` and `C_2`
  are the covariance matrices of distributions 1 and 2, respectively

  Args:
    dist1: A MultivariateNormalLinearOperator distribution.
    dist2: A MultivariateNormalLinearOperator distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.
  """
  def get_scale_operator(d: MultivariateNormalLinearOperatorLike):
    """Helper to get the `scale` of type `LinearOperator`."""
    if isinstance(d, MultivariateNormalLinearOperator):
      return d.scale_operator
    assert isinstance(d, tfd.MultivariateNormalLinearOperator)
    return d.scale

  def _has_diagonal_scale(d: MultivariateNormalLinearOperatorLike) -> bool:
    """Helper to get the property `has_diagonal_scale` of a distribution."""
    if isinstance(d, MultivariateNormalLinearOperator):
      return d.has_diagonal_scale
    assert isinstance(d, tfd.MultivariateNormalLinearOperator)
    # For a TFP distribution, perform a brute-force comparison between the
    # diagonal part of the scale and the dense matrix
    return jnp.alltrue(jnp.equal(
        jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(d.scale.diag_part()),
        d.scale.to_dense()))

  def squared_frobenius_norm(x: Array) -> Array:
    """Helper to get the squared Frobenius norm of a matrix."""
    return jnp.sum(jnp.square(x), axis=[-2, -1])

  num_dims = tuple(dist1.event_shape)[-1]  # `tuple` needed for TFP distrib.
  if num_dims != tuple(dist2.event_shape)[-1]:
    raise ValueError(f'Both multivariate normal distributions must have the '
                     f'same `event_shape`, but they have {num_dims} and '
                     f'{tuple(dist2.event_shape)[-1]} dimensions.')

  # Calculation is based on:
  # https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/distributions/mvn_linear_operator.py#L384
  # If C_1 = AA', C_2 = BB', then
  #   tr[inv(C_2) C_1] = ||inv(B) A||_F^2
  # where ||.||_F^2 is the squared Frobenius norm.
  scale1 = get_scale_operator(dist1)
  scale2 = get_scale_operator(dist2)
  if _has_diagonal_scale(dist1) and _has_diagonal_scale(dist2):
    b_inv_a = jnp.expand_dims(dist1.stddev() / dist2.stddev(), axis=-1)
  else:
    b_inv_a = scale2.solve(scale1.to_dense())
  diff_lob_abs_det = scale2.log_abs_determinant() - scale1.log_abs_determinant()
  diff_mean_expanded = jnp.expand_dims(dist2.mean() - dist1.mean(), axis=-1)
  kl_divergence = (
      diff_lob_abs_det +
      0.5 * (-num_dims +
             squared_frobenius_norm(b_inv_a) +
             squared_frobenius_norm(scale2.solve(diff_mean_expanded))))
  return jnp.broadcast_to(
      kl_divergence,
      jax.lax.broadcast_shapes(
          tuple(dist1.batch_shape), tuple(dist2.batch_shape)))


# Register the KL functions with TFP
tfd.RegisterKL(
    MultivariateNormalLinearOperator,
    MultivariateNormalLinearOperator)(
        _kl_divergence_mvn_linear_op_mvn_linear_op)
tfd.RegisterKL(
    MultivariateNormalLinearOperator,
    MultivariateNormalLinearOperator.equiv_tfp_cls)(
        _kl_divergence_mvn_linear_op_mvn_linear_op)
tfd.RegisterKL(
    MultivariateNormalLinearOperator.equiv_tfp_cls,
    MultivariateNormalLinearOperator)(
        _kl_divergence_mvn_linear_op_mvn_linear_op)
