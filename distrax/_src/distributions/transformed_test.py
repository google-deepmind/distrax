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
"""Tests for `transformed.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import block
from distrax._src.bijectors import lambda_bijector
from distrax._src.bijectors import masked_coupling
from distrax._src.bijectors import scalar_affine
from distrax._src.bijectors import sigmoid
from distrax._src.distributions import normal
from distrax._src.distributions import transformed
from distrax._src.utils import conversion
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def _with_additional_parameters(params, all_named_parameters):
  """Convenience function for appending a cartesian product of parameters."""
  for name, param in params:
    for named_params in all_named_parameters:
      yield (f'{named_params[0]}; {name}',) + named_params[1:] + (param,)


def _with_base_dists(*all_named_parameters):
  """Partial of _with_additional_parameters to specify distrax and tfp base."""
  base_dists = (
      ('tfp_base', tfd.Normal),
      ('distrax_base', normal.Normal),
  )
  return _with_additional_parameters(base_dists, all_named_parameters)


class TransformedTest(parameterized.TestCase):

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
    bijector = tfb.Scale(2)
    dist = transformed.Transformed(base, bijector)
    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)

    assert dist.event_shape == tfp_dist.event_shape

  @parameterized.named_parameters(
      ('tfp_normal, tfp_scale',
       lambda: tfd.Normal(0, 1), lambda: tfb.Scale(2)),
      ('tfp_normal, tfp_shift',
       lambda: tfd.Normal(0, 1), lambda: tfb.Shift(3.0)),
      ('tfp_normal, tfp_tanh',
       lambda: tfd.Normal(0, 1), tfb.Tanh),
  )
  def test_dtype_is_consistent_with_tfp(self, dist_fn, bijector_fn):
    base = dist_fn()
    bijector = bijector_fn()
    dist = transformed.Transformed(base, bijector)
    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)
    assert dist.dtype == tfp_dist.dtype

  @chex.all_variants
  @parameterized.named_parameters(
      ('tfp_normal, dx_lambda_scale', lambda: tfd.Normal(0, 1),
       lambda: lambda x: x * 2, jnp.float32),
      ('tfp_normal, dx_lambda_shift', lambda: tfd.Normal(0, 1),
       lambda: lambda x: x + 3.0, jnp.float32),
      ('tfp_normal, dx_lambda_tanh', lambda: tfd.Normal(0, 1),
       lambda: jnp.tanh, jnp.float32),
  )
  def test_dtype_is_as_expected(self, dist_fn, bijector_fn, expected_dtype):
    base = dist_fn()
    bijector = bijector_fn()
    dist = transformed.Transformed(base, bijector)
    sample = self.variant(dist.sample)(seed=self.seed)
    assert dist.dtype == sample.dtype
    assert dist.dtype == expected_dtype

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
    bijector = tfb.Scale(2)
    dist = transformed.Transformed(base, bijector)
    def sample_fn(seed, sample_shape):
      return dist.sample(seed=seed, sample_shape=sample_shape)
    samples = self.variant(sample_fn, ignore_argnums=(1,), static_argnums=1)(
        self.seed, sample_shape)

    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)
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
    bijector = tfb.Scale(2)
    dist = transformed.Transformed(base, bijector)
    actual = self.variant(dist.log_prob)(value)

    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)
    expected = tfp_dist.log_prob(value)
    np.testing.assert_array_equal(actual, expected)

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
    bijector = tfb.Scale(2)
    dist = transformed.Transformed(base, bijector)
    actual = self.variant(dist.prob)(value)

    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)
    expected = tfp_dist.prob(value)
    np.testing.assert_array_equal(actual, expected)

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
    bijector = tfb.Tanh()
    dist = transformed.Transformed(base, bijector)
    def sample_and_log_prob_fn(seed, sample_shape):
      return dist.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
    samples, log_prob = self.variant(
        sample_and_log_prob_fn, ignore_argnums=(1,), static_argnums=(1,))(
            self.seed, sample_shape)
    expected_samples = bijector.forward(
        base.sample(seed=self.seed, sample_shape=sample_shape))

    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)
    tfp_samples = tfp_dist.sample(seed=self.seed, sample_shape=sample_shape)
    tfp_log_prob = tfp_dist.log_prob(samples)

    chex.assert_equal_shape([samples, tfp_samples])
    np.testing.assert_allclose(log_prob, tfp_log_prob, rtol=1e-2)
    np.testing.assert_allclose(samples, expected_samples, rtol=1e-2)

  @parameterized.named_parameters(
      ('1d-batched bijector,  unbatched sample', (2,), ()),
      ('1d-batched bijector, 1d-batched sample', (2,), (4,)),
      ('1d-batched bijector, 2d-batched sample', (2,), (4, 5)),
      ('2d-batched bijector,  unbatched sample', (5, 2), ()),
      ('2d-batched bijector, 1d-batched sample', (5, 2), (4,)),
      ('2d-batched bijector, 2d-batched sample', (5, 2), (4, 5)),
      ('3d-batched bijector,  unbatched sample', (7, 5, 2), ()),
      ('3d-batched bijector, 1d-batched sample', (7, 5, 2), (4,)),
      ('3d-batched bijector, 2d-batched sample', (7, 5, 2), (4, 5)),
  )
  def test_batched_bijector_shapes(self, batch_shape, sample_shape):
    base = tfd.MultivariateNormalDiag(jnp.zeros(3), jnp.ones(3))
    bijector = block.Block(tfb.Scale(jnp.ones(batch_shape + (3,))), 1)
    dist = transformed.Transformed(base, bijector)

    with self.subTest('batch_shape'):
      chex.assert_equal(dist.batch_shape, batch_shape)

    with self.subTest('sample.shape'):
      sample = dist.sample(seed=self.seed, sample_shape=sample_shape)
      chex.assert_equal(sample.shape, sample_shape + batch_shape + (3,))

    with self.subTest('sample_and_log_prob sample.shape'):
      sample, log_prob = dist.sample_and_log_prob(
          seed=self.seed, sample_shape=sample_shape)
      chex.assert_equal(sample.shape, sample_shape + batch_shape + (3,))

    with self.subTest('sample_and_log_prob log_prob.shape'):
      sample, log_prob = dist.sample_and_log_prob(
          seed=self.seed, sample_shape=sample_shape)
      chex.assert_equal(log_prob.shape, sample_shape + batch_shape)

    with self.subTest('sample_and_log_prob log_prob value'):
      sample, log_prob = dist.sample_and_log_prob(
          seed=self.seed, sample_shape=sample_shape)
      np.testing.assert_allclose(log_prob, dist.log_prob(sample))

  @chex.all_variants
  @parameterized.named_parameters(
      ('Scale-scalar-unbatched', tfb.Scale, 1, (), 3),
      ('Scale-scalar-batched', tfb.Scale, 1, (), (2, 3)),
      ('Scale-vector-unbatched', tfb.Scale, 1, 3, 3),
      ('Scale-vector-batched', tfb.Scale, 1, 3, (2, 3)),
      ('Scale-batched-unbatched', tfb.Scale, 1, (2, 3), 3),
      ('Scale-batched-batched', tfb.Scale, 1, (2, 3), (2, 3)),
      ('Matvec-vector-unbatched', tfb.ScaleMatvecDiag, 0, 3, 3),
      ('Matvec-vector-batched', tfb.ScaleMatvecDiag, 0, 3, (2, 3)),
      ('Matvec-batched-unbatched', tfb.ScaleMatvecDiag, 0, (2, 3), 3),
      ('Matvec-batched-batched', tfb.ScaleMatvecDiag, 0, (2, 3), (2, 3)),
  )
  def test_batched_bijector_against_tfp(
      self, bijector_fn, block_ndims, bijector_shape, params_shape):

    base = tfd.MultivariateNormalDiag(
        jnp.zeros(params_shape), jnp.ones(params_shape))

    tfp_bijector = bijector_fn(jnp.ones(bijector_shape))
    dx_bijector = block.Block(tfp_bijector, block_ndims)

    dx_dist = transformed.Transformed(base, dx_bijector)
    tfp_dist = tfd.TransformedDistribution(
        conversion.to_tfp(base), tfp_bijector)

    with self.subTest('event_shape property matches TFP'):
      np.testing.assert_equal(dx_dist.event_shape, tfp_dist.event_shape)

    with self.subTest('sample shape matches TFP'):
      dx_sample = self.variant(dx_dist.sample)(seed=self.seed)
      tfp_sample = self.variant(tfp_dist.sample)(seed=self.seed)
      chex.assert_equal_shape([dx_sample, tfp_sample])

    with self.subTest('log_prob(dx_sample) matches TFP'):
      dx_logp_dx = self.variant(dx_dist.log_prob)(dx_sample)
      tfp_logp_dx = self.variant(tfp_dist.log_prob)(dx_sample)
      np.testing.assert_allclose(dx_logp_dx, tfp_logp_dx, rtol=1e-2)

    with self.subTest('log_prob(tfp_sample) matches TFP'):
      dx_logp_tfp = self.variant(dx_dist.log_prob)(tfp_sample)
      tfp_logp_tfp = self.variant(tfp_dist.log_prob)(tfp_sample)
      np.testing.assert_allclose(dx_logp_tfp, tfp_logp_tfp, rtol=1e-2)

    with self.subTest('sample/lp shape is self-consistent'):
      second_sample, log_prob = self.variant(dx_dist.sample_and_log_prob)(
          seed=self.seed)
      chex.assert_equal_shape([dx_sample, second_sample])
      chex.assert_equal_shape([dx_logp_dx, log_prob])

  # These should all fail because the bijector's event_ndims is incompatible
  # with the base distribution's event_shape.
  @parameterized.named_parameters(
      ('scalar', 0),
      ('matrix', 2),
      ('3-tensor', 3),
  )
  def test_raises_on_incorrect_shape(self, block_dims):
    base = tfd.MultivariateNormalDiag(jnp.zeros((2, 3)), jnp.ones((2, 3)))
    scalar_bijector = tfb.Scale(jnp.ones((1, 2, 3)))
    block_bijector = block.Block(scalar_bijector, block_dims)
    with self.assertRaises(ValueError):
      transformed.Transformed(base, block_bijector)

  @chex.all_variants
  def test_bijector_that_assumes_batch_dimensions(self):
    # Create a Haiku conditioner that assumes a single batch dimension.
    def forward(x):
      network = hk.Sequential([hk.Flatten(preserve_dims=1), hk.Linear(3)])
      return network(x)
    init, apply = hk.transform(forward)
    params = init(self.seed, jnp.ones((2, 3)))
    conditioner = functools.partial(apply, params, self.seed)

    bijector = masked_coupling.MaskedCoupling(
        jnp.ones(3) > 0, conditioner, tfb.Scale)

    base = tfd.MultivariateNormalDiag(jnp.zeros((2, 3)), jnp.ones((2, 3)))
    dist = transformed.Transformed(base, bijector)
    # Exercise the trace-based functions
    assert dist.batch_shape == (2,)
    assert dist.event_shape == (3,)
    assert dist.dtype == jnp.float32
    sample = self.variant(dist.sample)(seed=self.seed)
    assert sample.dtype == dist.dtype
    self.variant(dist.log_prob)(sample)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(_with_base_dists(
      ('entropy', 'entropy', 0., 1.),
      ('mean', 'mean', 0, 1),
      ('mean from list params', 'mean', [-1, 1], [1, 2]),
      ('mode', 'mode', 0, 1),
  ))
  def test_method(self, function_string, mu, sigma, base_dist):
    base = base_dist(mu, sigma)
    bijector = tfb.Scale(2)
    dist = transformed.Transformed(base, bijector)
    tfp_dist = tfd.TransformedDistribution(conversion.to_tfp(base), bijector)

    np.testing.assert_allclose(
        self.variant(getattr(dist, function_string))(),
        getattr(tfp_dist, function_string)())

  @chex.all_variants
  @parameterized.named_parameters(_with_base_dists(
      ('int16', np.array([0, 0], dtype=np.int16)),
      ('int32', np.array([0, 0], dtype=np.int32)),
      ('int64', np.array([0, 0], dtype=np.int64)),
  ))
  def test_integer_inputs(self, inputs, base_dist):
    base = base_dist(jnp.zeros_like(inputs, dtype=jnp.float32),
                     jnp.ones_like(inputs, dtype=jnp.float32))
    bijector = scalar_affine.ScalarAffine(shift=0.0)
    dist = transformed.Transformed(base, bijector)

    log_prob = self.variant(dist.log_prob)(inputs)

    standard_normal_log_prob_of_zero = -0.9189385
    expected_log_prob = jnp.full_like(
        inputs, standard_normal_log_prob_of_zero, dtype=jnp.float32)

    np.testing.assert_array_equal(log_prob, expected_log_prob)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('kl distrax_to_distrax', 'distrax_to_distrax'),
      ('kl distrax_to_tfp', 'distrax_to_tfp'),
      ('kl tfp_to_distrax', 'tfp_to_distrax'))
  def test_kl_divergence(self, mode_string):
    base_dist1 = tfd.Normal([0.1, 0.5, 0.9], [0.1, 1.1, 2.5])
    base_dist2 = tfd.Normal(-0.1, 1.5)
    bij_tfp1 = tfb.Identity()
    bij_tfp2 = tfb.Identity()
    bij_distrax1 = bij_tfp1
    bij_distrax2 = lambda_bijector.Lambda(lambda x: x)
    tfp_dist1 = tfd.TransformedDistribution(base_dist1, bij_tfp1)
    tfp_dist2 = tfd.TransformedDistribution(base_dist2, bij_tfp2)
    distrax_dist1 = transformed.Transformed(base_dist1, bij_distrax1)
    distrax_dist2 = transformed.Transformed(base_dist2, bij_distrax2)

    expected_result_fwd = base_dist1.kl_divergence(base_dist2)
    expected_result_inv = base_dist2.kl_divergence(base_dist1)

    distrax_fn1 = self.variant(distrax_dist1.kl_divergence)
    distrax_fn2 = self.variant(distrax_dist2.kl_divergence)

    if mode_string == 'distrax_to_distrax':
      result_fwd = distrax_fn1(distrax_dist2)
      result_inv = distrax_fn2(distrax_dist1)
    elif mode_string == 'distrax_to_tfp':
      result_fwd = distrax_fn1(tfp_dist2)
      result_inv = distrax_fn2(tfp_dist1)
    elif mode_string == 'tfp_to_distrax':
      result_fwd = tfp_dist1.kl_divergence(distrax_dist2)
      result_inv = tfp_dist2.kl_divergence(distrax_dist1)

    np.testing.assert_allclose(result_fwd, expected_result_fwd, rtol=1e-2)
    np.testing.assert_allclose(result_inv, expected_result_inv, rtol=1e-2)

  @chex.all_variants(with_pmap=False)
  def test_kl_divergence_on_same_instance_of_distrax_bijector(self):
    base_dist1 = tfd.Normal([0.1, 0.5, 0.9], [0.1, 1.1, 2.5])
    base_dist2 = tfd.Normal(-0.1, 1.5)
    bij_distrax = sigmoid.Sigmoid()
    distrax_dist1 = transformed.Transformed(base_dist1, bij_distrax)
    distrax_dist2 = transformed.Transformed(base_dist2, bij_distrax)
    expected_result_fwd = base_dist1.kl_divergence(base_dist2)
    expected_result_inv = base_dist2.kl_divergence(base_dist1)
    result_fwd = self.variant(distrax_dist1.kl_divergence)(distrax_dist2)
    result_inv = self.variant(distrax_dist2.kl_divergence)(distrax_dist1)
    np.testing.assert_allclose(result_fwd, expected_result_fwd, rtol=1e-2)
    np.testing.assert_allclose(result_inv, expected_result_inv, rtol=1e-2)

  def test_kl_divergence_raises_on_event_shape(self):
    base_dist1 = tfd.MultivariateNormalDiag([0.1, 0.5, 0.9], [0.1, 1.1, 2.5])
    base_dist2 = tfd.Normal(-0.1, 1.5)
    bij1 = block.Block(lambda_bijector.Lambda(lambda x: x), ndims=1)
    bij2 = lambda_bijector.Lambda(lambda x: x)
    distrax_dist1 = transformed.Transformed(base_dist1, bij1)
    distrax_dist2 = transformed.Transformed(base_dist2, bij2)
    with self.assertRaises(ValueError):
      distrax_dist1.kl_divergence(distrax_dist2)

  def test_kl_divergence_raises_on_different_bijectors(self):
    base_dist1 = tfd.Normal([0.1, 0.5, 0.9], [0.1, 1.1, 2.5])
    base_dist2 = tfd.Normal(-0.1, 1.5)
    bij1 = lambda_bijector.Lambda(lambda x: x)
    bij2 = sigmoid.Sigmoid()
    distrax_dist1 = transformed.Transformed(base_dist1, bij1)
    distrax_dist2 = transformed.Transformed(base_dist2, bij2)
    with self.assertRaises(NotImplementedError):
      distrax_dist1.kl_divergence(distrax_dist2)

  def test_jittable(self):
    @jax.jit
    def f(x, d):
      return d.log_prob(x)

    base = normal.Normal(0, 1)
    bijector = scalar_affine.ScalarAffine(0, 1)
    dist = transformed.Transformed(base, bijector)
    x = np.zeros(())
    f(x, dist)

if __name__ == '__main__':
  absltest.main()
