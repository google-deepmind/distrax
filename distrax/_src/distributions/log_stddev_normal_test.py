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
"""Tests for `log_stddev_normal.py`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import log_stddev_normal as lsn
from distrax._src.distributions import normal
from distrax._src.utils import compat
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp


kl_module = tfp.distributions.kullback_leibler


class LogStddevNormalTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.zeros((4,)), np.zeros((4,)), np.zeros((4,))),
      (np.zeros(()), np.zeros((4,)), np.zeros((4,))),
      (np.zeros((4,)), np.zeros(()), np.zeros((4,))),
  )
  def test_log_scale_property(self, mean, log_stddev, expected):
    dist = lsn.LogStddevNormal(mean, log_stddev)
    self.assertEqual(dist.log_scale.shape, expected.shape)
    np.testing.assert_allclose(dist.log_scale, expected, atol=1e-4)

  @parameterized.parameters(
      (0.0, 1.0),
      (4.0, 10.0),
  )
  def test_sampling_scalar(self, mean, stddev):
    log_stddev = np.log(stddev)
    dist = lsn.LogStddevNormal(mean, log_stddev)

    num_samples = 1000000
    prng_key = jax.random.PRNGKey(1331)
    samples = dist.sample(seed=prng_key, sample_shape=num_samples)
    chex.assert_shape(samples, (num_samples,))
    np.testing.assert_allclose(jnp.mean(samples), mean, atol=4e-2)
    np.testing.assert_allclose(jnp.std(samples), stddev, atol=4e-2)

  @parameterized.parameters(
      ([3, 4], [1.5, 2.5]),
      ([0, 1, 0, 1, 10], [0.1, 0.5, 1.0, 5.0, 10.0]))
  def test_sampling_vector(self, mean, stddev):
    mean = np.array(mean)
    log_stddev = np.log(stddev)
    self.assertEqual(mean.shape, log_stddev.shape)
    dist = lsn.LogStddevNormal(mean, log_stddev)

    num_samples = 1000000
    prng_key = jax.random.PRNGKey(1331)
    samples = dist.sample(seed=prng_key, sample_shape=num_samples)
    chex.assert_shape(samples, (num_samples,) + mean.shape)
    np.testing.assert_allclose(jnp.mean(samples, axis=0), mean, atol=4e-2)
    np.testing.assert_allclose(jnp.std(samples, axis=0), stddev, atol=4e-2)

  def test_sampling_batched(self):
    means = np.array([[3.0, 4.0], [-5, 48.0], [58, 64.0]])
    stddevs = np.array([[1, 2], [2, 4], [4, 8]])
    log_stddevs = np.log(stddevs)
    dist = lsn.LogStddevNormal(means, log_stddevs)

    num_samples = 1000000
    prng_key = jax.random.PRNGKey(1331)
    samples = dist.sample(seed=prng_key, sample_shape=num_samples)
    # output shape is [num_samples] + means.shape
    chex.assert_shape(samples, (num_samples, 3, 2))
    np.testing.assert_allclose(jnp.mean(samples, axis=0), means, atol=4e-2)
    np.testing.assert_allclose(jnp.std(samples, axis=0), stddevs, atol=4e-2)

  def test_sampling_batched_custom_dim(self):
    means = np.array([[3.0, 4.0], [-5, 48.0], [58, 64.0]])
    stddevs = np.array([[1, 2], [2, 4], [4, 8]])
    log_stddevs = np.log(stddevs)
    dist = lsn.LogStddevNormal(means, log_stddevs)

    num_samples = 1000000
    prng_key = jax.random.PRNGKey(1331)
    samples = dist.sample(seed=prng_key, sample_shape=num_samples)
    chex.assert_shape(samples, (num_samples, 3, 2))
    np.testing.assert_allclose(jnp.mean(samples, axis=0), means, atol=4e-2)
    np.testing.assert_allclose(jnp.std(samples, axis=0), stddevs, atol=4e-2)

  @chex.all_variants
  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('float64', jnp.float64))
  def test_sample_dtype(self, dtype):
    with compat.enable_x64(dtype.dtype.itemsize == 8):
      dist = lsn.LogStddevNormal(
          loc=jnp.zeros((), dtype), log_scale=jnp.zeros((), dtype))
      # pyrefly: ignore[missing-attribute]
      samples = self.variant(dist.sample)(seed=jax.random.PRNGKey(0))
      self.assertEqual(samples.dtype, dist.dtype)
      chex.assert_type(samples, dtype)

  def test_kl_versus_normal(self):
    loc, scale = jnp.array([2.0]), jnp.array([2.0])
    log_scale = jnp.log(scale)
    lsn_prior = lsn.LogStddevNormal(jnp.array([0.0]), jnp.array([0.0]))
    n_prior = normal.Normal(jnp.array([0.0]), jnp.array([1.0]))
    lsn_dist = lsn.LogStddevNormal(loc, log_scale)
    n_dist = normal.Normal(loc, scale)

    kl1 = tfp.distributions.kl_divergence(lsn_dist, lsn_prior)
    kl2 = tfp.distributions.kl_divergence(n_dist, lsn_prior)
    kl3 = tfp.distributions.kl_divergence(n_dist, n_prior)
    np.testing.assert_allclose(kl2, kl1)
    np.testing.assert_allclose(kl3, kl2)
    np.testing.assert_allclose(kl1, kl3)

  @parameterized.named_parameters(
      ('float32_small', jnp.float32, -50.0, None),
      ('float32_large', jnp.float32, 50.0, None),
      ('float32_control', jnp.float32, 0.0, None),
      ('float16_small', jnp.float16, -10.0, None),
      ('float16_large', jnp.float16, 10.0, None),
      ('float16_control', jnp.float16, 0.0, None),
      ('capped_small', jnp.float32, -50.0, 2.0),
      ('capped_large', jnp.float32, 50.0, 2.0),
      ('capped_control', jnp.float32, 0.0, 2.0),
  )
  def test_kl_values_and_gradients(self, dtype, log_scale, max_scale):
    def effective_scale(raw):
      if max_scale is None:
        return raw, np.ones_like(raw)
      log_max = np.log(max_scale)
      return (
          log_max - np.logaddexp(0.0, log_max - raw),
          1.0 / (1.0 + np.exp(raw - log_max)),
      )

    log_scale1 = jnp.array(
        [[log_scale - 0.5], [log_scale], [log_scale + 0.5]], dtype
    )
    log_scale2 = jnp.array(log_scale, dtype)
    effective2, chain2 = effective_scale(np.asarray(log_scale2, np.float64))
    loc1 = jnp.array(np.exp(effective2) * np.array([-0.5, 0.0, 0.5]), dtype)
    loc2 = jnp.array(0.0, dtype)

    # Independent float64 reference uses the actual rounded input parameters.
    effective1, chain1 = effective_scale(np.asarray(log_scale1, np.float64))
    delta = effective1 - effective2
    z = (np.asarray(loc1, np.float64) - float(loc2)) * np.exp(-effective2)
    expected = 0.5 * (np.square(z) + np.expm1(2.0 * delta)) - delta
    expected_grads = (
        3.0 * z,
        3.0 * np.expm1(2.0 * delta) * chain1,
        -3.0 * np.sum(z),
        np.sum(-np.square(z) - np.exp(2.0 * delta) + 1.0) * chain2,
    )

    def kl(loc1, log_scale1, loc2, log_scale2):
      dist1 = lsn.LogStddevNormal(loc1, log_scale1, max_scale=max_scale)
      dist2 = lsn.LogStddevNormal(loc2, log_scale2, max_scale=max_scale)
      return dist1.kl_divergence(dist2)

    grad_fn = jax.grad(lambda *args: jnp.sum(kl(*args)), argnums=(0, 1, 2, 3))
    tolerance = 2e-2 if dtype == jnp.float16 else 3e-5
    for values_fn, gradients_fn in (
        (kl, grad_fn),
        (jax.jit(kl), jax.jit(grad_fn)),
    ):
      args = (loc1, log_scale1, loc2, log_scale2)
      actual = values_fn(*args)
      self.assertEqual(actual.dtype, dtype)
      np.testing.assert_allclose(
          actual, expected, rtol=tolerance, atol=tolerance
      )
      grads = gradients_fn(*args)
      for i, (actual_grad, wanted_grad) in enumerate(
          zip(grads, expected_grads)
      ):
        self.assertEqual(actual_grad.dtype, dtype)
        # Compare mean derivatives in scale-relative units so both very small
        # and very large finite gradients receive meaningful tolerances.
        normalized_grad = np.asarray(actual_grad, np.float64)
        if i in (0, 2):
          normalized_grad *= np.exp(effective2)
        np.testing.assert_allclose(
            normalized_grad, wanted_grad, rtol=tolerance, atol=tolerance
        )

  @parameterized.named_parameters(
      ('float32_small', jnp.float32, -50.0),
      ('float32_control', jnp.float32, 0.0),
      ('float32_large', jnp.float32, 50.0),
      ('float16_subnormal_scale', jnp.float16, -12.0),
      ('float16_small', jnp.float16, -10.0),
      ('float16_control', jnp.float16, 0.0),
      ('float16_large', jnp.float16, 10.0),
  )
  def test_kl_identical_log_scales(self, dtype, log_scale):
    def kl(loc1, log_scale1, loc2, log_scale2):
      dist1 = lsn.LogStddevNormal(loc1, log_scale1)
      dist2 = lsn.LogStddevNormal(loc2, log_scale2)
      return dist1.kl_divergence(dist2)

    args = (
        jnp.array(0.0, dtype),
        jnp.array(log_scale, dtype),
        jnp.array(0.0, dtype),
        jnp.array(log_scale, dtype),
    )
    grad_fn = jax.value_and_grad(kl, argnums=(0, 1, 2, 3))
    for value_and_grad in (
        grad_fn,
        jax.jit(grad_fn),
    ):
      value, grads = value_and_grad(*args)
      for actual in (value, *grads):
        self.assertEqual(actual.dtype, dtype)
        np.testing.assert_array_equal(actual, 0.0)

  # pylint:disable=protected-access
  def test_custom_kl_registered(self):
    # Check that our custom KL is registered inside the TFP dispatch table.
    dist_pair = (lsn.LogStddevNormal, lsn.LogStddevNormal)
    self.assertEqual(kl_module._DIVERGENCES[dist_pair],
                     lsn._kl_logstddevnormal_logstddevnormal)

  @mock.patch.dict(
      kl_module._DIVERGENCES,
      {(lsn.LogStddevNormal,
        lsn.LogStddevNormal): lambda *args, **kwargs: 42})
  def test_calling_custom_kl(self):
    # Check that the dispatch of tfp.kl_divergence actually goes to the
    # table we checked for above.
    dist_a = lsn.LogStddevNormal(jnp.array([0.0]), jnp.array([0.0]))
    dist_b = lsn.LogStddevNormal(jnp.array([0.0]), jnp.array([0.0]))
    self.assertEqual(tfp.distributions.kl_divergence(dist_a, dist_b), 42)
  # pylint:enable=protected-access

  def test_jitable(self):
    @jax.jit
    def jitted_function(event, dist):
      return dist.log_prob(event)

    dist = lsn.LogStddevNormal(np.array([0.0]), np.array([0.0]))
    event = dist.sample(seed=jax.random.PRNGKey(0))
    jitted_function(event, dist)

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    loc = jnp.array(np.random.randn(3, 4, 5))
    log_scale = jnp.array(np.random.randn(3, 4, 5))
    dist = lsn.LogStddevNormal(loc=loc, log_scale=log_scale)
    sliced_dist = dist[slice_]
    np.testing.assert_allclose(sliced_dist.mean(), loc[slice_], rtol=1e-3)
    np.testing.assert_allclose(
        sliced_dist.log_scale, log_scale[slice_], rtol=1e-3)
    self.assertIsInstance(sliced_dist, lsn.LogStddevNormal)

  def test_slice_different_parameterization(self):
    loc = jnp.array(np.random.randn(4))
    log_scale = jnp.array(np.random.randn(3, 4))
    dist = lsn.LogStddevNormal(loc=loc, log_scale=log_scale)
    np.testing.assert_allclose(
        dist[0].mean(), loc, rtol=1e-3)  # Not slicing loc.
    np.testing.assert_allclose(dist[0].log_scale, log_scale[0], rtol=1e-3)

if __name__ == '__main__':
  absltest.main()
