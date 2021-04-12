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
"""Tests for `conversion.py`."""

import sys

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from distrax._src.bijectors.bijector import Bijector
from distrax._src.bijectors.rational_quadratic_spline import RationalQuadraticSpline
from distrax._src.bijectors.tanh import Tanh
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.normal import Normal
from distrax._src.distributions.transformed import Transformed
from distrax._src.utils import conversion
import jax
from jax.config import config as jax_config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
FLAGS = flags.FLAGS
flags.DEFINE_bool('test_jax_enable_x64', False,
                  'Whether to enable double precision for tests.')


def setUpModule():
  if not FLAGS.is_parsed():
    FLAGS(sys.argv, known_only=True)
  if FLAGS['test_jax_enable_x64'].value:
    jax_config.update('jax_enable_x64', True)


class AsBijectorTest(parameterized.TestCase):

  def test_num_bins_attr_of_rational_quadratic_spline(self):
    num_bins = 4
    bijector = RationalQuadraticSpline(
        jnp.zeros((3 * num_bins + 1,)),
        range_min=0.,
        range_max=1.)
    wrapped_bijector = conversion.as_bijector(bijector)
    assert isinstance(wrapped_bijector, RationalQuadraticSpline)
    self.assertIs(wrapped_bijector, bijector)
    # Access the `num_bins` attribute of a wrapped RationalQuadraticSpline.
    np.testing.assert_equal(wrapped_bijector.num_bins, num_bins)

  def test_on_tfp_bijector(self):
    inputs = jnp.array([0., 1.])
    bijector = tfb.Exp()
    wrapped_bijector = conversion.as_bijector(bijector)
    assert isinstance(wrapped_bijector, Bijector)
    np.testing.assert_array_almost_equal(
        wrapped_bijector.forward(inputs),
        bijector.forward(inputs))


class AsDistributionTest(parameterized.TestCase):

  def test_loc_attr_of_normal(self):
    dist = Normal(loc=0., scale=1.)
    wrapped_dist = conversion.as_distribution(dist)
    assert isinstance(wrapped_dist, Normal)
    self.assertIs(wrapped_dist, dist)
    # Access the `loc` attribute of a wrapped Normal.
    np.testing.assert_almost_equal(wrapped_dist.loc, 0.)

  def test_num_categories_attr_of_categorical(self):
    dist = Categorical(logits=jnp.array([0., 0., 0.]))
    wrapped_dist = conversion.as_distribution(dist)
    assert isinstance(wrapped_dist, Categorical)
    self.assertIs(wrapped_dist, dist)
    # Access the `num_categories` attribute of a wrapped Categorical.
    np.testing.assert_equal(wrapped_dist.num_categories, 3)

  def test_attrs_of_transformed_distribution(self):
    dist = Transformed(Normal(loc=0., scale=1.), bijector=lambda x: x)
    wrapped_dist = conversion.as_distribution(dist)
    assert isinstance(wrapped_dist, Transformed)
    self.assertIs(wrapped_dist, dist)
    # Access the `distribution` attribute of a wrapped Transformed.
    assert isinstance(wrapped_dist.distribution, Normal)
    # Access the `loc` attribute of a transformed Normal within a wrapped
    # Transformed.
    np.testing.assert_almost_equal(wrapped_dist.distribution.loc, 0.)

  def test_on_tfp_distribution(self):
    dist = tfd.Normal(loc=0., scale=1.)
    wrapped_dist = conversion.as_distribution(dist)
    assert isinstance(wrapped_dist, tfd.Normal)
    assert isinstance(wrapped_dist, Distribution)
    # Access the `loc` attribute of a wrapped Normal.
    np.testing.assert_almost_equal(wrapped_dist.loc, 0.)


class ToTfpTest(parameterized.TestCase):

  def test_on_distrax_distribution(self):
    dist = Normal(loc=0., scale=1.)
    wrapped_dist = conversion.to_tfp(dist)
    assert isinstance(wrapped_dist, Normal)
    # Access the `loc` attribute of a wrapped Normal.
    np.testing.assert_almost_equal(wrapped_dist.loc, 0.)

  def test_on_distrax_bijector(self):
    bij = Tanh()
    wrapped_bij = conversion.to_tfp(bij)
    assert isinstance(wrapped_bij, Tanh)
    # Call the `forward` attribute of a wrapped Tanh.
    np.testing.assert_equal(
        wrapped_bij.forward(np.zeros(())), bij.forward(np.zeros(())))

  def test_on_tfp_distribution(self):
    dist = tfd.Normal(0., 1.)
    wrapped_dist = conversion.to_tfp(dist)
    self.assertIs(wrapped_dist, dist)

  def test_on_tfp_bijector(self):
    bij = tfb.Exp()
    wrapped_bij = conversion.to_tfp(bij)
    self.assertIs(wrapped_bij, bij)


class AsFloatArrayTest(parameterized.TestCase):

  @parameterized.parameters(0, 0.1)
  def test_on_valid_scalar(self, x):
    y = conversion.as_float_array(x)
    self.assertIsInstance(y, jnp.ndarray)
    self.assertEqual(
        y.dtype, jnp.float64 if jax.config.x64_enabled else jnp.float32)

  @parameterized.parameters(True, 1j)
  def test_on_invalid_scalar(self, x):
    with self.assertRaises(ValueError):
      conversion.as_float_array(x)

  @parameterized.parameters(
      float, jnp.float_, jnp.float16, jnp.float32, jnp.float64, jnp.bfloat16)
  def test_on_float_array(self, dtype):
    x = jnp.zeros([], dtype)
    y = conversion.as_float_array(x)
    self.assertIs(y, x)

  @parameterized.parameters(
      int, jnp.int_, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
      jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64)
  def test_on_int_array(self, dtype):
    x = jnp.zeros([], dtype)
    y = conversion.as_float_array(x)
    self.assertIsInstance(y, jnp.ndarray)
    self.assertEqual(
        y.dtype, jnp.float64 if jax.config.x64_enabled else jnp.float32)

  @parameterized.parameters(
      bool, jnp.bool_, complex, jnp.complex_, jnp.complex64, jnp.complex128)
  def test_on_invalid_array(self, dtype):
    x = jnp.zeros([], dtype)
    with self.assertRaises(ValueError):
      conversion.as_float_array(x)


if __name__ == '__main__':
  absltest.main()
