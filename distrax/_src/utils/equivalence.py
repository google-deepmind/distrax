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
"""Utility functions for testing equivalence between TFP and Distrax."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl.testing import parameterized

import chex
from distrax._src.distributions import distribution
import jax
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array


def get_tfp_equiv(distrax_cls):
  """Returns the TFP equivalent of a Distrax class.

  Args:
    distrax_cls: The Distrax class or the name of the class as a string.

  Returns:
    The equivalent TFP class if found, else `None`.

  Raises:
    ValueError:
      If `distrax_cls` is neither a Distrax class nor a string.

  #### Examples
  from distrax import normal
  from distrax import tfp_utils

  # equivalent to tfd.normal.Normal(0, 1)
  tfp_utils.get_tfp_equiv(normal.Normal)(0, 1)

  # equivalent to tfd.normal.Normal(0, 1)
  tfp_utils.get_tfp_equiv("Normal")(0, 1)

  """

  if isinstance(distrax_cls, str):
    tfp_class_str = distrax_cls
  elif issubclass(distrax_cls, distribution.Distribution):
    tfp_class_str = distrax_cls.__name__
  else:
    raise ValueError(
        'distrax_cls must be the class object or the name of the class object'
        ' as a string'
    )

  if hasattr(tfd, tfp_class_str):
    return getattr(tfd, tfp_class_str)
  else:
    return None


class EquivalenceTest(parameterized.TestCase):
  """Provides comparison assertions for TFP and Distrax distributions."""

  def setUp(self):
    super().setUp()
    self.tfp_cls = None

  def _init_distr_cls(self, distrax_cls: type(distribution.Distribution)):
    self.key = jax.random.PRNGKey(1234)
    self.distrax_cls = distrax_cls
    if hasattr(distrax_cls, 'equiv_tfp_cls'):
      self.tfp_cls = distrax_cls.equiv_tfp_cls
    else:
      self.tfp_cls = get_tfp_equiv(distrax_cls)

  def assertion_fn(self, **kwargs) -> Callable[[Array, Array], None]:
    def fn(x: Array, y: Array) -> None:
      np.testing.assert_allclose(x, y, **kwargs)
    return fn

  def _test_attribute(
      self,
      attribute_string: str,
      dist_args: Tuple[Any, ...] = (),
      dist_kwargs: Optional[Dict[str, Any]] = None,
      tfp_dist_args: Optional[Tuple[Any, ...]] = None,
      tfp_dist_kwargs: Optional[Dict[str, Any]] = None,
      call_args: Tuple[Any, ...] = (),
      call_kwargs: Optional[Dict[str, Any]] = None,
      assertion_fn: Callable[[Any, Any], None] = np.testing.assert_allclose):
    """Asserts equivalence of TFP and Distrax attributes.

    Given a potentially callable attribute as a string, compares the attribute
    values among Distrax and TFP implementations.

    Args:
      attribute_string: An attribute or a method of a Distrax/TFP
        distribution, provided as a string.
      dist_args: Arguments to be passed to Distrax constructor as *dist_args.
      dist_kwargs: Keyword arguments to be passed to Distrax constructor as
        **dist_kwargs.
      tfp_dist_args: Arguments to be passed to TFP constructor as
        *tfp_dist_args. If None, defaults to `dist_args`.
      tfp_dist_kwargs: Keyword arguments to be passed to TFP constructor as
        **tfp_dist_kwargs. If None, defaults to `dist_kwargs`.
      call_args: Arguments to be passed to Distrax and TFP methods as
        *call_args.
      call_kwargs: Keyword arguments to be passed to Distrax and TFP
        methods as **call_kwargs.
      assertion_fn: Assertion function to be called to compare
        Distrax and TFP methods/attributes.
    """

    if dist_kwargs is None:
      dist_kwargs = {}
    if call_kwargs is None:
      call_kwargs = {}
    if tfp_dist_args is None:
      tfp_dist_args = dist_args
    if tfp_dist_kwargs is None:
      tfp_dist_kwargs = dist_kwargs

    dist = self.distrax_cls(*dist_args, **dist_kwargs)
    tfp_dist = self.tfp_cls(*tfp_dist_args, **tfp_dist_kwargs)

    if callable(getattr(dist, attribute_string)):
      distrax_fn = getattr(dist, attribute_string)
      tfp_fn = getattr(tfp_dist, attribute_string)
      if hasattr(self, 'variant'):
        distrax_fn = self.variant(distrax_fn)
      assertion_fn(distrax_fn(*call_args, **call_kwargs),
                   tfp_fn(*call_args, **call_kwargs))
    else:
      assertion_fn(getattr(dist, attribute_string),
                   getattr(tfp_dist, attribute_string))

  def _test_event_shape(self, dist_args, dist_kwargs,
                        tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests event shape."""
    self._test_attribute('event_shape', dist_args, dist_kwargs,
                         tfp_dist_args, tfp_dist_kwargs)

  def _test_batch_shape(self, dist_args, dist_kwargs,
                        tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests batch shape."""
    self._test_attribute('batch_shape', dist_args, dist_kwargs,
                         tfp_dist_args, tfp_dist_kwargs)

  def _test_prob(self, dist_args, dist_kwargs, value,
                 tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests prob."""
    self._test_attribute('prob', dist_args, dist_kwargs, tfp_dist_args,
                         tfp_dist_kwargs, (value,))

  def _test_log_prob(self, dist_args, dist_kwargs, value,
                     tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests log prob."""
    assertion_fn = functools.partial(np.testing.assert_allclose, rtol=1e-2)
    self._test_attribute(
        'log_prob', dist_args, dist_kwargs, tfp_dist_args, tfp_dist_kwargs,
        (value,), assertion_fn=assertion_fn)

  def _test_cdf(self, dist_args, dist_kwargs, value,
                tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests CDF."""
    self._test_attribute('cdf', dist_args, dist_kwargs,
                         tfp_dist_args, tfp_dist_kwargs, (value,))

  def _test_log_cdf(self, dist_args, dist_kwargs, value,
                    tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests log CDF."""
    self._test_attribute('log_cdf', dist_args, dist_kwargs,
                         tfp_dist_args, tfp_dist_kwargs, (value,))

  def _test_sample_shape(self, dist_args, dist_kwargs, sample_shape,
                         tfp_dist_args=None, tfp_dist_kwargs=None):
    """Tests sample shape."""
    if tfp_dist_args is None:
      tfp_dist_args = dist_args
    if tfp_dist_kwargs is None:
      tfp_dist_kwargs = dist_kwargs
    dist = self.distrax_cls(*dist_args, **dist_kwargs)

    def sample_fn(key, sample_shape=sample_shape):
      return dist.sample(seed=key, sample_shape=sample_shape)

    if hasattr(self, 'variant'):
      sample_fn = self.variant(sample_fn)
    samples = sample_fn(self.key)

    tfp_dist = self.tfp_cls(*tfp_dist_args, **tfp_dist_kwargs)
    tfp_samples = tfp_dist.sample(sample_shape=sample_shape,
                                  seed=self.key)
    chex.assert_equal_shape([samples, tfp_samples])

  def _test_sample_and_log_prob(
      self,
      dist_args: Tuple[Any, ...] = (),
      dist_kwargs: Optional[Dict[str, Any]] = None,
      tfp_dist_args: Optional[Tuple[Any, ...]] = None,
      tfp_dist_kwargs: Optional[Dict[str, Any]] = None,
      sample_shape: Union[int, Tuple[int, ...]] = (),
      assertion_fn: Callable[[Any, Any], None] = np.testing.assert_allclose):
    """Tests sample and log prob."""
    if tfp_dist_args is None:
      tfp_dist_args = dist_args
    if tfp_dist_kwargs is None:
      tfp_dist_kwargs = dist_kwargs
    dist = self.distrax_cls(*dist_args, **dist_kwargs)
    log_prob_fn = dist.log_prob

    def sample_and_log_prob_fn(key):
      return dist.sample_and_log_prob(seed=key, sample_shape=sample_shape)

    if hasattr(self, 'variant'):
      sample_and_log_prob_fn = self.variant(sample_and_log_prob_fn)
      log_prob_fn = self.variant(dist.log_prob)
    samples, log_prob = sample_and_log_prob_fn(self.key)

    tfp_dist = self.tfp_cls(*tfp_dist_args, **tfp_dist_kwargs)
    tfp_samples = tfp_dist.sample(sample_shape=sample_shape,
                                  seed=self.key)
    tfp_log_prob = tfp_dist.log_prob(samples)

    chex.assert_equal_shape([samples, tfp_samples])
    assertion_fn(log_prob, tfp_log_prob)
    assertion_fn(log_prob, log_prob_fn(samples))

  def _test_with_two_distributions(
      self,
      attribute_string: str,
      mode_string: str = 'distrax_to_distrax',
      dist1_args: Tuple[Any, ...] = (),
      dist1_kwargs: Optional[Dict[str, Any]] = None,
      dist2_args: Tuple[Any, ...] = (),
      dist2_kwargs: Optional[Dict[str, Any]] = None,
      tfp_dist1_args: Tuple[Any, ...] = (),
      tfp_dist1_kwargs: Optional[Dict[str, Any]] = None,
      tfp_dist2_args: Tuple[Any, ...] = (),
      tfp_dist2_kwargs: Optional[Dict[str, Any]] = None,
      assertion_fn: Callable[[Any, Any], None] = np.testing.assert_allclose):
    """Asserts equivalence of TFP and Distrax methods that compare two distribs.

    This checks that the methods `D(dist1 || dist2)` and `D(dist2 || dist1)`
    give the same results as their TFP counterparts, where `D` is typically the
    KL divergence or the cross-entropy.

    Args:
      attribute_string: The method attribute, provided as a string.
      mode_string: string, must be one of the following:
        - If "distrax_to_distrax", this method verifies the values of
          `D(dist1 || dist2)` and `D(dist2 || dist1)`, where both `dist1` and
          `dist2` are Distrax distributions.
        - If "distrax_to_tfp", this method verifies the values of
          `D(dist1 || tfp_dist2)` and `D(dist2 || tfp_dist1)`.
        - If "tfp_to_distrax", this method verifies the values of
          `D(tfp_dist1 || dist2)` and `D(tfp_dist2 || dist1)`.
      dist1_args: Arguments to be passed to Distrax constructor as *dist_args
        for the first distribution
      dist1_kwargs: Keyword arguments to be passed to Distrax constructor as
        **dist_kwargs for the first distribution.
      dist2_args: Arguments to be passed to Distrax constructor as *dist_args
        for the second distribution.
      dist2_kwargs: Keyword arguments to be passed to Distrax constructor as
        **dist_kwargs for the second distribution.
      tfp_dist1_args: Arguments to be passed to TFP constructor as
        *tfp_dist_args for the first distribution. If None, defaults to
        `dist1_args`.
      tfp_dist1_kwargs: Keyword arguments to be passed to TFP constructor as
        **tfp_dist_kwargs for the first distribution. If None, defaults to
        `dist1_kwargs`.
      tfp_dist2_args: Arguments to be passed to TFP constructor as
        *tfp_dist_args for the second distribution. If None, defaults to
        `dist2_args`.
      tfp_dist2_kwargs: Keyword arguments to be passed to TFP constructor as
        **tfp_dist_kwargs for the second distribution. If None, defaults to
        `dist2_kwargs`.
      assertion_fn: Assertion function to be called to compare Distrax and TFP
        function values.
    """
    dist1_kwargs = {} if dist1_kwargs is None else dist1_kwargs
    dist2_kwargs = {} if dist2_kwargs is None else dist2_kwargs

    if tfp_dist1_args is None:
      tfp_dist1_args = dist1_args
    if tfp_dist1_kwargs is None:
      tfp_dist1_kwargs = dist1_kwargs
    if tfp_dist2_args is None:
      tfp_dist2_args = dist2_args
    if tfp_dist2_kwargs is None:
      tfp_dist2_kwargs = dist2_kwargs

    dist1 = self.distrax_cls(*dist1_args, **dist1_kwargs)
    tfp_dist1 = self.tfp_cls(*tfp_dist1_args, **tfp_dist1_kwargs)
    dist2 = self.distrax_cls(*dist2_args, **dist2_kwargs)
    tfp_dist2 = self.tfp_cls(*tfp_dist2_args, **tfp_dist2_kwargs)

    tfp_comp_dist1_dist2 = getattr(tfp_dist1, attribute_string)(tfp_dist2)
    tfp_comp_dist2_dist1 = getattr(tfp_dist2, attribute_string)(tfp_dist1)

    distrax_fn_1 = getattr(dist1, attribute_string)
    distrax_fn_2 = getattr(dist2, attribute_string)
    if hasattr(self, 'variant'):
      distrax_fn_1 = self.variant(distrax_fn_1)
      distrax_fn_2 = self.variant(distrax_fn_2)

    if mode_string == 'distrax_to_distrax':
      comp_dist1_dist2 = distrax_fn_1(dist2)
      comp_dist2_dist1 = distrax_fn_2(dist1)
    elif mode_string == 'distrax_to_tfp':
      comp_dist1_dist2 = distrax_fn_1(tfp_dist2)
      comp_dist2_dist1 = distrax_fn_2(tfp_dist1)
    elif mode_string == 'tfp_to_distrax':
      comp_dist1_dist2 = getattr(tfp_dist1, attribute_string)(dist2)
      comp_dist2_dist1 = getattr(tfp_dist2, attribute_string)(dist1)
    else:
      raise ValueError(
          f'`mode_string` should be one of the following: '
          f'"distrax_to_distrax", "distrax_to_tfp", or "tfp_to_distrax", '
          f'but it is "{mode_string}".')

    assertion_fn(comp_dist1_dist2, tfp_comp_dist1_dist2)
    assertion_fn(comp_dist2_dist1, tfp_comp_dist2_dist1)

  def _test_jittable(
      self,
      dist_args: Tuple[Any, ...] = (),
      dist_kwargs: Optional[Dict[str, Any]] = None,
      assertion_fn: Callable[[Any, Any], None] = np.testing.assert_allclose):
    """Tests that the distribution can be passed to a jitted function."""
    dist_kwargs = dist_kwargs or {}

    @jax.jit
    def jitted_function(event, dist):
      return dist.log_prob(event)

    dist = self.distrax_cls(*dist_args, **dist_kwargs)
    event = dist.sample(seed=self.key)
    log_prob = dist.log_prob(event)
    jitted_log_prob = jitted_function(event, dist)
    assertion_fn(jitted_log_prob, log_prob)

  def _test_raises_error(
      self,
      dist_args: Tuple[Any, ...] = (),
      dist_kwargs: Optional[Dict[str, Any]] = None,
      error_type=AssertionError):
    """Tests that the instantiation of the distribution raises an error."""
    dist_kwargs = dist_kwargs or {}
    try:
      with self.assertRaises(error_type):
        self.distrax_cls(*dist_args, **dist_kwargs)
    except ValueError:
      # For forward compatibility with Chex (it will raise AssertionErrors
      # instead of ValueErrors in the new version) .
      # TODO(iukemaev): remove after the new version of Chex is released.
      pass
