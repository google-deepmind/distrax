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
"""Tests for `inverse.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import bijector as base_bijector
from distrax._src.bijectors import inverse
from distrax._src.bijectors import scalar_affine
from distrax._src.distributions import normal
from distrax._src.distributions import transformed
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


RTOL = 1e-2


def _with_additional_parameters(params, all_named_parameters):
  """Convenience function for appending a cartesian product of parameters."""
  for name, param in params:
    for named_params in all_named_parameters:
      yield (f'{named_params[0]}; {name}',) + named_params[1:] + (param,)


def _with_base_dists(*all_named_parameters):
  """Partial of _with_additional_parameters to specify distrax and TFP base."""
  base_dists = (
      ('tfp_base', tfd.Normal),
      ('distrax_base', normal.Normal),
  )
  return _with_additional_parameters(base_dists, all_named_parameters)


class InverseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  def test_properties(self):
    bijector = inverse.Inverse(tfb.Scale(2))
    assert isinstance(bijector.bijector, base_bijector.Bijector)

  @parameterized.named_parameters(_with_base_dists(
      ('1d std normal', 0, 1),
      ('2d std normal', np.zeros(2), np.ones(2)),
      ('broadcasted loc', 0, np.ones(3)),
      ('broadcasted scale', np.ones(3), 1),
  ))
  def test_event_shape(self, mu, sigma, base_dist):
    base = base_dist(mu, sigma)
    bijector = inverse.Inverse(tfb.Scale(2))
    dist = transformed.Transformed(base, bijector)

    tfp_bijector = tfb.Invert(tfb.Scale(2))
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)

    assert dist.event_shape == tfp_dist.event_shape

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('1d std normal, no shape', 0, 1, ()),
      ('1d std normal, int shape', 0, 1, 1),
      ('1d std normal, 1-tuple shape', 0, 1, (1,)),
      ('1d std normal, 2-tuple shape', 0, 1, (2, 2)),
      ('2d std normal, no shape', np.zeros(2), np.ones(2), ()),
      ('2d std normal, int shape', [0, 0], [1, 1], 1),
      ('2d std normal, 1-tuple shape', np.zeros(2), np.ones(2), (1,)),
      ('2d std normal, 2-tuple shape', [0, 0], [1, 1], (2, 2)),
      ('rank 2 std normal, 2-tuple shape', np.zeros(
          (3, 2)), np.ones((3, 2)), (2, 2)),
      ('broadcasted loc', 0, np.ones(3), (2, 2)),
      ('broadcasted scale', np.ones(3), 1, ()),
  ))
  def test_sample_shape(self, mu, sigma, sample_shape, base_dist):
    base = base_dist(mu, sigma)
    bijector = inverse.Inverse(tfb.Scale(2))
    dist = transformed.Transformed(base, bijector)
    def sample_fn(seed, sample_shape):
      return dist.sample(seed=seed, sample_shape=sample_shape)
    samples = self.variant(sample_fn, ignore_argnums=(1,), static_argnums=1)(
        self.seed, sample_shape)

    tfp_bijector = tfb.Invert(tfb.Scale(2))
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    tfp_samples = tfp_dist.sample(sample_shape=sample_shape, seed=self.seed)

    chex.assert_equal_shape([samples, tfp_samples])

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('1d dist, 1d value', 0, 1, 1.),
      ('1d dist, 2d value', 0., 1., np.array([1., 2.])),
      ('2d dist, 1d value', np.zeros(2), np.ones(2), 1.),
      ('2d broadcasted dist, 1d value', np.zeros(2), 1, 1.),
      ('2d dist, 2d value', np.zeros(2), np.ones(2), np.array([1., 2.])),
      ('1d dist, 1d value, edge case', 0, 1, 200.),
  ))
  def test_log_prob(self, mu, sigma, value, base_dist):
    base = base_dist(mu, sigma)
    bijector = inverse.Inverse(tfb.Scale(2))
    dist = transformed.Transformed(base, bijector)
    actual = self.variant(dist.log_prob)(value)

    tfp_bijector = tfb.Invert(tfb.Scale(2))
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    expected = tfp_dist.log_prob(value)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('1d dist, 1d value', 0, 1, 1.),
      ('1d dist, 2d value', 0., 1., np.array([1., 2.])),
      ('2d dist, 1d value', np.zeros(2), np.ones(2), 1.),
      ('2d broadcasted dist, 1d value', np.zeros(2), 1, 1.),
      ('2d dist, 2d value', np.zeros(2), np.ones(2), np.array([1., 2.])),
      ('1d dist, 1d value, edge case', 0, 1, 200.),
  ))
  def test_prob(self, mu, sigma, value, base_dist):
    base = base_dist(mu, sigma)
    bijector = inverse.Inverse(tfb.Scale(2))
    dist = transformed.Transformed(base, bijector)
    actual = self.variant(dist.prob)(value)

    tfp_bijector = tfb.Invert(tfb.Scale(2))
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    expected = tfp_dist.prob(value)
    np.testing.assert_allclose(actual, expected, atol=1e-9)

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('1d std normal, no shape', 0, 1, ()),
      ('1d std normal, int shape', 0, 1, 1),
      ('1d std normal, 1-tuple shape', 0, 1, (1,)),
      ('1d std normal, 2-tuple shape', 0, 1, (2, 2)),
      ('2d std normal, no shape', np.zeros(2), np.ones(2), ()),
      ('2d std normal, int shape', [0, 0], [1, 1], 1),
      ('2d std normal, 1-tuple shape', np.zeros(2), np.ones(2), (1,)),
      ('2d std normal, 2-tuple shape', [0, 0], [1, 1], (2, 2)),
      ('rank 2 std normal, 2-tuple shape', np.zeros(
          (3, 2)), np.ones((3, 2)), (2, 2)),
      ('broadcasted loc', 0, np.ones(3), (2, 2)),
      ('broadcasted scale', np.ones(3), 1, ()),
  ))
  def test_sample_and_log_prob(self, mu, sigma, sample_shape, base_dist):
    base = base_dist(mu, sigma)
    bijector = inverse.Inverse(tfb.Scale(2))
    dist = transformed.Transformed(base, bijector)
    def sample_and_log_prob_fn(seed, sample_shape):
      return dist.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
    samples, log_prob = self.variant(
        sample_and_log_prob_fn, ignore_argnums=(1,), static_argnums=(1,))(
            self.seed, sample_shape)
    expected_samples = bijector.forward(
        base.sample(seed=self.seed, sample_shape=sample_shape))

    tfp_bijector = tfb.Invert(tfb.Scale(2))
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    tfp_samples = tfp_dist.sample(seed=self.seed, sample_shape=sample_shape)
    tfp_log_prob = tfp_dist.log_prob(samples)

    chex.assert_equal_shape([samples, tfp_samples])
    np.testing.assert_allclose(log_prob, tfp_log_prob, rtol=RTOL)
    np.testing.assert_allclose(samples, expected_samples, rtol=RTOL)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(_with_base_dists(
      ('entropy', 'entropy', 0., 1.),
      ('mean', 'mean', 0, 1),
      ('mean from list params', 'mean', [-1, 1], [1, 2]),
      ('mode', 'mode', 0, 1),
  ))
  def test_method(self, function_string, mu, sigma, base_dist):
    base = base_dist(mu, sigma)
    bijector = inverse.Inverse(tfb.Scale(2))
    dist = transformed.Transformed(base, bijector)

    tfp_bijector = tfb.Invert(tfb.Scale(2))
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)

    np.testing.assert_allclose(
        self.variant(getattr(dist, function_string))(),
        getattr(tfp_dist, function_string)())

  @chex.all_variants
  @parameterized.named_parameters(
      ('int16', np.array([1, 2], dtype=np.int16)),
      ('int32', np.array([1, 2], dtype=np.int32)),
      ('int64', np.array([1, 2], dtype=np.int64)),
  )
  def test_integer_inputs(self, inputs):
    bijector = inverse.Inverse(scalar_affine.ScalarAffine(shift=1.0))
    output, log_det = self.variant(bijector.forward_and_log_det)(inputs)

    expected_out = jnp.array(inputs, dtype=jnp.float32) - 1.0
    expected_log_det = jnp.zeros_like(inputs, dtype=jnp.float32)

    np.testing.assert_array_equal(output, expected_out)
    np.testing.assert_array_equal(log_det, expected_log_det)

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = inverse.Inverse(scalar_affine.ScalarAffine(0, 1))
    x = np.zeros(())
    f(x, bijector)

if __name__ == '__main__':
  absltest.main()
