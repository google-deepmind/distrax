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
"""Tests for `lambda_bijector.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import lambda_bijector
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


class LambdaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = jax.random.PRNGKey(1234)

  @parameterized.named_parameters(_with_base_dists(
      ('1d std normal', 0, 1),
      ('2d std normal', np.zeros(2), np.ones(2)),
      ('broadcasted loc', 0, np.ones(3)),
      ('broadcasted scale', np.ones(3), 1),
  ))
  def test_event_shape(self, mu, sigma, base_dist):
    base = base_dist(mu, sigma)
    bijector = lambda_bijector.Lambda(jnp.tanh)
    dist = transformed.Transformed(base, bijector)

    tfp_bijector = tfb.Tanh()
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)

    assert dist.event_shape == tfp_dist.event_shape

  def test_raises_on_both_none(self):
    with self.assertRaises(ValueError):
      lambda_bijector.Lambda(forward=None, inverse=None)

  def test_raises_on_log_det_without_event_ndims(self):
    with self.assertRaises(ValueError):
      lambda_bijector.Lambda(
          forward=lambda x: x,
          forward_log_det_jacobian=lambda x: jnp.zeros_like(x[:-1]),
          event_ndims_in=None)

  @parameterized.named_parameters(
      ('event_ndims_in', 0, None),
      ('event_ndims_out', None, 0),
      ('event_ndims_in and event_ndims_out', 0, 0),
  )
  def test_raises_on_event_ndims_without_log_det(self, ndims_in, ndims_out):
    with self.assertRaises(ValueError):
      lambda_bijector.Lambda(
          forward=lambda x: x,
          event_ndims_in=ndims_in,
          event_ndims_out=ndims_out)

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
    bijector = lambda_bijector.Lambda(jnp.tanh)
    dist = transformed.Transformed(base, bijector)
    def sample_fn(seed, sample_shape):
      return dist.sample(seed=seed, sample_shape=sample_shape)
    samples = self.variant(sample_fn, ignore_argnums=(1,), static_argnums=1)(
        self.seed, sample_shape)

    tfp_bijector = tfb.Tanh()
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    tfp_samples = tfp_dist.sample(sample_shape=sample_shape,
                                  seed=self.seed)

    chex.assert_equal_shape([samples, tfp_samples])

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('1d dist, 1d value', 0, 1, 0.5),
      ('1d dist, 2d value', 0., 1., np.array([0.25, 0.5])),
      ('2d dist, 1d value', np.zeros(2), np.ones(2), 0.5),
      ('2d broadcasted dist, 1d value', np.zeros(2), 1, 0.5),
      ('2d dist, 2d value', np.zeros(2), np.ones(2), np.array([0.25, 0.5])),
      ('1d dist, 1d value, edge case', 0, 1, 0.99),
  ))
  def test_log_prob(self, mu, sigma, value, base_dist):
    base = base_dist(mu, sigma)
    bijector = lambda_bijector.Lambda(jnp.tanh)
    dist = transformed.Transformed(base, bijector)
    actual = self.variant(dist.log_prob)(value)

    tfp_bijector = tfb.Tanh()
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    expected = tfp_dist.log_prob(value)

    np.testing.assert_allclose(actual, expected, rtol=RTOL)

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('1d dist, 1d value', 0, 1, 0.5),
      ('1d dist, 2d value', 0., 1., np.array([0.25, 0.5])),
      ('2d dist, 1d value', np.zeros(2), np.ones(2), 0.5),
      ('2d broadcasted dist, 1d value', np.zeros(2), 1, 0.5),
      ('2d dist, 2d value', np.zeros(2), np.ones(2), np.array([0.25, 0.5])),
      ('1d dist, 1d value, edge case', 0, 1, 0.99),
  ))
  def test_prob(self, mu, sigma, value, base_dist):
    base = base_dist(mu, sigma)
    bijector = lambda_bijector.Lambda(jnp.tanh)
    dist = transformed.Transformed(base, bijector)
    actual = self.variant(dist.prob)(value)

    tfp_bijector = tfb.Tanh()
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)
    expected = tfp_dist.prob(value)

    np.testing.assert_allclose(actual, expected, rtol=RTOL)

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
    bijector = lambda_bijector.Lambda(lambda x: 10 * jnp.tanh(0.1 * x))
    dist = transformed.Transformed(base, bijector)
    def sample_and_log_prob_fn(seed, sample_shape):
      return dist.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
    samples, log_prob = self.variant(
        sample_and_log_prob_fn, ignore_argnums=(1,), static_argnums=(1,))(
            self.seed, sample_shape)
    expected_samples = bijector.forward(
        base.sample(seed=self.seed, sample_shape=sample_shape))

    tfp_bijector = tfb.Chain([tfb.Scale(10), tfb.Tanh(), tfb.Scale(0.1)])
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
    bijector = lambda_bijector.Lambda(lambda x: x + 3)
    dist = transformed.Transformed(base, bijector)

    tfp_bijector = tfb.Shift(3)
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)

    np.testing.assert_allclose(self.variant(getattr(dist, function_string))(),
                               getattr(tfp_dist, function_string)(), rtol=RTOL)

  @chex.all_variants(with_jit=False)  # no need to jit function transformations
  @parameterized.named_parameters(
      ('identity', lambda x: x, tfb.Identity),
      ('tanh', jnp.tanh, tfb.Tanh),
      ('scale', lambda x: 3.0 * x, lambda: tfb.Scale(3.0)),
      ('shift', lambda x: x + 2.0, lambda: tfb.Shift(2.0)),
      ('exp', jnp.exp, tfb.Exp),
      ('softplus', lambda x: jnp.log1p(jnp.exp(x)), tfb.Softplus),
      ('square', jnp.square, tfb.Square),
  )
  def test_log_dets(self, lambda_bjct, tfp_bijector_fn):
    bijector = lambda_bijector.Lambda(lambda_bjct)
    tfp_bijector = tfp_bijector_fn()

    x = np.array([0.05, 0.3, 0.45], dtype=np.float32)
    fldj = tfp_bijector.forward_log_det_jacobian(x, event_ndims=0)
    fldj_ = self.variant(bijector.forward_log_det_jacobian)(x)
    np.testing.assert_allclose(fldj_, fldj, rtol=RTOL)

    y = bijector.forward(x)
    ildj = tfp_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    ildj_ = self.variant(bijector.inverse_log_det_jacobian)(y)
    np.testing.assert_allclose(ildj_, ildj, rtol=RTOL)

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('identity', lambda x: x, tfb.Identity),
      ('tanh', jnp.tanh, tfb.Tanh),
      ('scale', lambda x: 3.0 * x, lambda: tfb.Scale(3.0)),
      ('shift', lambda x: x + 2.0, lambda: tfb.Shift(2.0)),
      ('exp', jnp.exp, tfb.Exp),
      ('softplus', lambda x: jnp.log1p(jnp.exp(x)), tfb.Softplus),
      ('square', jnp.square, tfb.Square),
  ))
  def test_against_tfp_bijectors(
      self, lambda_bjct, tfp_bijector, base_dist):
    mu = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    sigma = np.array([0.5, 1.0, 2.5], dtype=np.float32)
    base = base_dist(mu, sigma)

    bijector = lambda_bijector.Lambda(lambda_bjct)
    dist = transformed.Transformed(base, bijector)
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector())

    y = np.array([0.05, 0.3, 0.95], dtype=np.float32)

    lp_y = tfp_dist.log_prob(y)
    lp_y_ = self.variant(dist.log_prob)(y)
    np.testing.assert_allclose(lp_y_, lp_y, rtol=RTOL)

    p_y = tfp_dist.prob(y)
    p_y_ = self.variant(dist.prob)(y)
    np.testing.assert_allclose(p_y_, p_y, rtol=RTOL)

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('identity', lambda x: x, tfb.Identity),
      ('tanh', jnp.tanh, tfb.Tanh),
      ('scale', lambda x: 3.0 * x, lambda: tfb.Scale(3.0)),
  ))
  def test_auto_lambda(
      self, forward_fn, tfp_bijector, base_dist):
    mu = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    sigma = np.array([0.5, 1.0, 2.5], dtype=np.float32)
    base = base_dist(mu, sigma)

    dist = transformed.Transformed(base, forward_fn)
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector())

    y = np.array([0.05, 0.3, 0.95], dtype=np.float32)

    lp_y = tfp_dist.log_prob(y)
    lp_y_ = self.variant(dist.log_prob)(y)
    np.testing.assert_allclose(lp_y_, lp_y, rtol=RTOL)

    p_y = tfp_dist.prob(y)
    p_y_ = self.variant(dist.prob)(y)
    np.testing.assert_allclose(p_y_, p_y, rtol=RTOL)

  def test_raises_on_invalid_input_shape(self):
    bij = lambda_bijector.Lambda(
        forward=lambda x: x,
        inverse=lambda y: y,
        forward_log_det_jacobian=lambda x: jnp.zeros_like(x[:-1]),
        inverse_log_det_jacobian=lambda y: jnp.zeros_like(y[:-1]),
        event_ndims_in=1)
    for fn in [bij.forward, bij.inverse,
               bij.forward_log_det_jacobian, bij.inverse_log_det_jacobian,
               bij.forward_and_log_det, bij.inverse_and_log_det]:
      with self.assertRaises(ValueError):
        fn(jnp.array(0))

  def test_jittable(self):
    @jax.jit
    def f(x, b):
      return b.forward(x)

    bijector = lambda_bijector.Lambda(lambda x: x)
    x = np.zeros(())
    f(x, bijector)


if __name__ == '__main__':
  absltest.main()
