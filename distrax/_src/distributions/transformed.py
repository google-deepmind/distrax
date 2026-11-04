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
"""Distribution representing a Bijector applied to a Distribution."""

from typing import Optional, Tuple, Union

from distrax._src.bijectors import bijector as bjct_base
from distrax._src.distributions import distribution as dist_base
from distrax._src.utils import conversion
import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

PRNGKey = dist_base.PRNGKey
Array = dist_base.Array
DistributionLike = dist_base.DistributionLike
BijectorLike = bjct_base.BijectorLike
EventT = dist_base.EventT


class Transformed(dist_base.Distribution):
  """Distribution of a random variable transformed by a bijective function.

  Let `X` be a continuous random variable and `Y = f(X)` be a random variable
  transformed by a differentiable bijection `f` (a "bijector"). Given the
  distribution of `X` (the "base distribution") and the bijector `f`, this class
  implements the distribution of `Y` (also known as the pushforward of the base
  distribution through `f`).

  The probability density of `Y` can be computed by:

  `log p(y) = log p(x) - log|det J(f)(x)|`

  where `p(x)` is the probability density of `X` (the "base density") and
  `J(f)(x)` is the Jacobian matrix of `f`, both evaluated at `x = f^{-1}(y)`.

  Sampling from a Transformed distribution involves two steps: sampling from the
  base distribution `x ~ p(x)` and then evaluating `y = f(x)`. The first step
  is agnostic to the possible batch dimensions of the bijector `f(x)`. For
  example:
  ```
    dist = distrax.Normal(loc=0., scale=1.)
    bij = distrax.ScalarAffine(shift=jnp.asarray([3., 3., 3.]))
    transformed_dist = distrax.Transformed(distribution=dist, bijector=bij)
    samples = transformed_dist.sample(seed=0, sample_shape=())
    print(samples)  # [2.7941577, 2.7941577, 2.7941577]
  ```

  Note: the `batch_shape`, `event_shape`, and `dtype` properties of the
  transformed distribution, as well as the `kl_divergence` method, are computed
  on-demand via JAX tracing when requested. This assumes that the `forward`
  function of the bijector is traceable; that is, it is a pure function that
  does not contain run-time branching. Functions that do not strictly meet this
  requirement can still be used, but we cannot guarantee that the shapes, dtype,
  and KL computations involving the transformed distribution can be correctly
  obtained.
  """

  equiv_tfp_cls = tfd.TransformedDistribution

  def __init__(self, distribution: DistributionLike, bijector: BijectorLike):
    """Initializes a Transformed distribution.

    Args:
      distribution: the base distribution. Can be either a Distrax distribution
        or a TFP distribution.
      bijector: a differentiable bijective transformation. Can be a Distrax
        bijector, a TFP bijector, or a callable to be wrapped by `Lambda`.
    """
    super().__init__()
    distribution = conversion.as_distribution(distribution)
    bijector = conversion.as_bijector(bijector)

    event_shape = distribution.event_shape
    # Check if event shape is a tuple of integers (i.e. not nested).
    if not (isinstance(event_shape, tuple) and
            all(isinstance(i, int) for i in event_shape)):
      raise ValueError(
          f"'Transformed' currently only supports distributions with Array "
          f"events (i.e. not nested). Received '{distribution.name}' with "
          f"event shape '{distribution.event_shape}'.")

    # pyrefly: ignore[missing-attribute]
    if len(event_shape) != bijector.event_ndims_in:
      raise ValueError(
          # pyrefly: ignore[missing-attribute]
          f"Base distribution '{distribution.name}' has event shape "
          f"{distribution.event_shape}, but bijector '{bijector.name}' expects "
          f"events to have {bijector.event_ndims_in} dimensions. Perhaps use "
          f"`distrax.Block` or `distrax.Independent`?")

    self._distribution = distribution
    self._bijector = bijector
    self._batch_shape = None
    self._event_shape = None
    self._dtype = None

  @property
  def distribution(self):
    """The base distribution."""
    return self._distribution

  @property
  def bijector(self):
    """The bijector representing the transformation."""
    return self._bijector

  def _infer_shapes_and_dtype(self):
    """Infer the batch shape, event shape, and dtype by tracing `forward`."""
    dummy_shape = self.distribution.batch_shape + self.distribution.event_shape
    dummy = jnp.zeros(dummy_shape, dtype=self.distribution.dtype)
    shape_dtype = jax.eval_shape(self.bijector.forward, dummy)

    self._dtype = shape_dtype.dtype

    if self.bijector.event_ndims_out == 0:
      self._event_shape = ()
      self._batch_shape = shape_dtype.shape
    else:
      # pylint: disable-next=invalid-unary-operand-type
      self._event_shape = shape_dtype.shape[-self.bijector.event_ndims_out:]
      # pylint: disable-next=invalid-unary-operand-type
      self._batch_shape = shape_dtype.shape[:-self.bijector.event_ndims_out]

  @property
  def dtype(self) -> jnp.dtype:
    """See `Distribution.dtype`."""
    if self._dtype is None:
      self._infer_shapes_and_dtype()
    assert self._dtype is not None  # By _infer_shapes_and_dtype()
    return self._dtype

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """See `Distribution.event_shape`."""
    if self._event_shape is None:
      self._infer_shapes_and_dtype()
    assert self._event_shape is not None  # By _infer_shapes_and_dtype()
    return self._event_shape

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """See `Distribution.batch_shape`."""
    if self._batch_shape is None:
      self._infer_shapes_and_dtype()
    assert self._batch_shape is not None  # By _infer_shapes_and_dtype()
    return self._batch_shape

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    x, ildj_y = self.bijector.inverse_and_log_det(value)
    lp_x = self.distribution.log_prob(x)
    lp_y = lp_x + ildj_y
    return lp_y

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    """Returns `n` samples."""
    x = self.distribution.sample(seed=key, sample_shape=n)
    y = jax.vmap(self.bijector.forward)(x)
    return y

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """Returns `n` samples and their log probs.

    This function is more efficient than calling `sample` and `log_prob`
    separately, because it uses only the forward methods of the bijector. It
    also works for bijectors that don't implement inverse methods.

    Args:
      key: PRNG key.
      n: Number of samples to generate.

    Returns:
      A tuple of `n` samples and their log probs.
    """
    x, lp_x = self.distribution.sample_and_log_prob(seed=key, sample_shape=n)
    y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x)
    lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
    return y, lp_y

  def mean(self) -> Array:
    """Calculates the mean."""
    if self.bijector.is_constant_jacobian:
      return self.bijector.forward(self.distribution.mean())
    else:
      raise NotImplementedError(
          "`mean` is not implemented for this transformed distribution, "
          "because its bijector's Jacobian is not known to be constant.")

  def mode(self) -> Array:
    """Calculates the mode."""
    if self.bijector.is_constant_log_det:
      return self.bijector.forward(self.distribution.mode())
    else:
      raise NotImplementedError(
          "`mode` is not implemented for this transformed distribution, "
          "because its bijector's Jacobian determinant is not known to be "
          "constant.")

  def entropy(  # pylint: disable=arguments-differ
      self,
      input_hint: Optional[Array] = None) -> Array:
    """Calculates the Shannon entropy (in Nats).

    Only works for bijectors with constant Jacobian determinant.

    Args:
      input_hint: an example sample from the base distribution, used to compute
        the constant forward log-determinant. If not specified, it is computed
        using a zero array of the shape and dtype of a sample from the base
        distribution.

    Returns:
      the entropy of the distribution.

    Raises:
      NotImplementedError: if bijector's Jacobian determinant is not known to be
                           constant.
    """
    if self.bijector.is_constant_log_det:
      if input_hint is None:
        shape = self.distribution.batch_shape + self.distribution.event_shape
        input_hint = jnp.zeros(shape, dtype=self.distribution.dtype)
      entropy = self.distribution.entropy()
      fldj = self.bijector.forward_log_det_jacobian(input_hint)
      return entropy + fldj
    else:
      raise NotImplementedError(
          "`entropy` is not implemented for this transformed distribution, "
          "because its bijector's Jacobian determinant is not known to be "
          "constant.")


def _jaxpr_constants(jaxpr):
  """Yields constants in a program and any nested equation programs."""
  yield from getattr(jaxpr, 'consts', ())
  program = getattr(jaxpr, 'jaxpr', jaxpr)
  for equation in program.eqns:
    is_jaxpr = lambda p: isinstance(p, (jax_core.Jaxpr, jax_core.ClosedJaxpr))
    for param in jax.tree_util.tree_leaves(equation.params, is_leaf=is_jaxpr):
      if isinstance(param, (jax_core.Jaxpr, jax_core.ClosedJaxpr)):
        yield from _jaxpr_constants(param)


def _same_constant(const1, const2) -> bool:
  """Proves equality of concrete representations or a shared binding."""
  if const1 is const2:
    return True
  # Distinct outer tracers cannot be compared at Python tracing time.
  if (isinstance(const1, jax.core.Tracer) or
      isinstance(const2, jax.core.Tracer)):
    return False
  try:
    dtype1 = getattr(const1, 'dtype', None)
    dtype2 = getattr(const2, 'dtype', None)
    if dtype1 != dtype2:
      return False
    if (dtype1 is not None and
        jax.dtypes.issubdtype(dtype1, jax.dtypes.extended)):
      if not jax.dtypes.issubdtype(dtype1, jax.dtypes.prng_key):
        return False
      # Preserve the PRNG implementation dtype before unwrapping key bits.
      with jax.ensure_compile_time_eval():
        const1 = jax.random.key_data(const1)
        const2 = jax.random.key_data(const2)
    array1, array2 = np.asarray(const1), np.asarray(const2)
  except (TypeError, ValueError):
    # Other extended dtypes may have no concrete NumPy representation.
    return False
  return (not array1.dtype.hasobject and not array2.dtype.hasobject and
          array1.dtype == array2.dtype and array1.shape == array2.shape and
          array1.tobytes(order='C') == array2.tobytes(order='C'))


def _same_closed_jaxpr(jaxpr1, jaxpr2) -> bool:
  """Checks program structure and proves equality of closed-over constants."""
  if str(jaxpr1.jaxpr) != str(jaxpr2.jaxpr):
    return False
  constants1 = list(_jaxpr_constants(jaxpr1))
  constants2 = list(_jaxpr_constants(jaxpr2))
  if len(constants1) != len(constants2):
    return False
  return all(_same_constant(c1, c2) for c1, c2 in zip(constants1, constants2))


def _kl_divergence_transformed_transformed(
    dist1: Union[Transformed, tfd.TransformedDistribution],
    dist2: Union[Transformed, tfd.TransformedDistribution],
    *unused_args,
    input_hint: Optional[Array] = None,
    **unused_kwargs,
) -> Array:
  """Obtains the KL divergence between two Transformed distributions.

  This computes the KL divergence between two Transformed distributions with the
  same bijector. If the two Transformed distributions do not have the same
  bijector, an error is raised. To determine if the bijectors are equal, this
  method proceeds as follows:
  - If both bijectors are the same instance of a Distrax bijector, then they are
    declared equal.
  - If not the same instance, we check if they are equal according to their
    `same_as` predicate.
  - Otherwise, the Jaxpr of each `forward` method and its closed-over constants
    are compared. Programs must have matching string representations, and their
    constants must be the same objects or have equal concrete shapes, dtypes,
    and byte representations (typed keys also require the same PRNG dtype).
    Distinct traced constants cannot prove equality.
  - Otherwise, the bijectors cannot be guaranteed to be equal and an error is
    raised.

  Args:
    dist1: A Transformed distribution.
    dist2: A Transformed distribution.
    input_hint: an example sample from the base distribution, used to trace the
      `forward` method. If not specified, it is computed using a zero array of
      the shape and dtype of a sample from the base distribution.

  Returns:
    Batchwise `KL(dist1 || dist2)`.

  Raises:
    NotImplementedError: If bijectors are not known to be equal.
    ValueError: If the base distributions do not have the same `event_shape`.
  """
  if dist1.distribution.event_shape != dist2.distribution.event_shape:
    raise ValueError(
        f"The two base distributions do not have the same event shape: "
        f"{dist1.distribution.event_shape} and "
        f"{dist2.distribution.event_shape}.")

  bij1 = conversion.as_bijector(dist1.bijector)  # conversion needed for TFP
  bij2 = conversion.as_bijector(dist2.bijector)

  # Check if the bijectors are different.
  if bij1 != bij2 and not bij1.same_as(bij2):
    if input_hint is None:
      input_hint = jnp.zeros(
          dist1.distribution.event_shape, dtype=dist1.distribution.dtype)
    jaxpr_bij1 = jax.make_jaxpr(bij1.forward)(input_hint)
    jaxpr_bij2 = jax.make_jaxpr(bij2.forward)(input_hint)
    if not _same_closed_jaxpr(jaxpr_bij1, jaxpr_bij2):
      raise NotImplementedError(
          f"The KL divergence cannot be obtained because it is not possible to "
          f"guarantee that the bijectors {dist1.bijector.name} and "
          f"{dist2.bijector.name} of the Transformed distributions are "
          f"equal. If possible, use the same instance of a Distrax bijector.")

  return dist1.distribution.kl_divergence(dist2.distribution)


# Register the KL functions with TFP.
tfd.RegisterKL(Transformed, Transformed)(_kl_divergence_transformed_transformed)
tfd.RegisterKL(Transformed.equiv_tfp_cls, Transformed)(
    _kl_divergence_transformed_transformed)
tfd.RegisterKL(Transformed, Transformed.equiv_tfp_cls)(
    _kl_divergence_transformed_transformed)
