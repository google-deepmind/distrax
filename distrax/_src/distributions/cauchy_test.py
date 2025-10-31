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
from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions import cauchy
from distrax._src.utils import equivalence
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class CauchyTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(cauchy.Cauchy)
    self.key = jax.random.PRNGKey(42)

  @parameterized.named_parameters(
      ('1d', (0.0, 1.0)),
      ('2d', (np.zeros(2), np.ones(2))),
      ('broadcast', (0.0, np.ones(3))),
  )
  def test_event_shape(self, dist_params):
    dist_params = tuple(jnp.asarray(p) for p in dist_params)
    super()._test_event_shape(dist_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d', (0.0, 1.0), ()),
      ('2d', (np.array([0.0, 1.0]), np.array([1.0, 2.0])), (100,)),
  )
  def test_sample_shape(self, dist_params, sample_shape):
    dist_params = tuple(jnp.asarray(p) for p in dist_params)
    super()._test_sample_shape(dist_params, dict(), sample_shape)

  @chex.all_variants
  @jax.numpy_rank_promotion('raise')
  @parameterized.named_parameters(
      ('1d', (0.0, 1.0), ()),
      ('2d', (np.array([0.0, 1.0]), np.array([1.0, 2.0])), (100,)),
  )
  def test_sample_and_log_prob(self, dist_params, sample_shape):
    dist_params = tuple(jnp.asarray(p) for p in dist_params)
    # High tolerance because sampling is unstable.
    super()._test_sample_and_log_prob(
        dist_args=dist_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn(rtol=1e-1),
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d', (0.0, 1.0), -1.0),
      (
          '2d',
          (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
          np.array([-1.0, 1.5]),
      ),
  )
  def test_method_with_input(self, dist_params, value):
    dist_params = tuple(jnp.asarray(p) for p in dist_params)
    value = jnp.asarray(value)
    # Test all methods that take `value` as input
    for method in [
        'log_prob',
        'prob',
        'cdf',
        'log_cdf',
        'survival_function',
        'log_survival_function',
    ]:
      with self.subTest(method):
        super()._test_attribute(
            attribute_string=method,
            dist_args=dist_params,
            dist_kwargs={},
            call_args=(value,),
            assertion_fn=self.assertion_fn(rtol=1e-2),
        )

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', (0.0, 1.0), 'entropy'),
      ('mean', (0.0, 1.0), 'mean'),
      ('variance', (0.0, 1.0), 'variance'),
      ('stddev', (0.0, 1.0), 'stddev'),
      ('mode', (0.0, 1.0), 'mode'),
  )
  def test_method(self, dist_params, function_string):
    dist_params = tuple(jnp.asarray(p) for p in dist_params)
    super()._test_attribute(
        function_string, dist_params, assertion_fn=self.assertion_fn(rtol=1e-2)
    )

  def test_jittable(self):
    # Overriding with a looser tolerance due to minor float differences.
    # Platforms like TPU and different OSs might have larger variance.
    super()._test_jittable(
        (0.0, 1.0), assertion_fn=self.assertion_fn(rtol=1e-2, atol=1e-2)
    )

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    loc = jnp.array(np.random.randn(3, 4, 5))
    scale = jnp.array(np.random.rand(3, 4, 5) + 0.1)  # Ensure scale is positive
    dist = self.distrax_cls(loc=loc, scale=scale)
    self.assertion_fn(rtol=1e-2)(dist[slice_].loc, loc[slice_])
    self.assertion_fn(rtol=1e-2)(dist[slice_].scale, scale[slice_])

  def test_slice_different_parameterization(self):
    loc = jnp.array(np.random.randn(4))
    scale = jnp.array(np.random.rand(3, 4) + 0.1)  # Ensure scale is positive
    dist = self.distrax_cls(loc=loc, scale=scale)
    self.assertion_fn(rtol=1e-2)(dist[0].loc, loc)  # Not slicing loc.
    self.assertion_fn(rtol=1e-2)(dist[0].scale, scale[0])

  def test_vmap_inputs(self):
    # Overriding because mean/variance are NaN and cause issues with
    # equivalence.EquivalenceTest's default vmap test.
    # We only test log_prob.
    def log_prob_sum(dist, x):
      return dist.log_prob(x).sum()

    dist = cauchy.Cauchy(
        jnp.arange(3 * 4 * 5).reshape((3, 4, 5)),
        jnp.ones((3, 4, 5)) * 0.1 + 1.0,
    )
    x = jnp.zeros((3, 4, 5))

    with self.subTest('no vmap'):
      actual = log_prob_sum(dist, x)
      expected = dist.log_prob(x).sum()
      self.assertion_fn(rtol=1e-2, atol=1e-2)(actual, expected)

    with self.subTest('axis=0'):
      actual = jax.vmap(log_prob_sum, in_axes=0)(dist, x)
      expected = dist.log_prob(x).sum(axis=(1, 2))
      self.assertion_fn(rtol=1e-2, atol=1e-2)(actual, expected)

    with self.subTest('axis=1'):
      actual = jax.vmap(log_prob_sum, in_axes=1)(dist, x)
      expected = dist.log_prob(x).sum(axis=(0, 2))
      self.assertion_fn(rtol=1e-2, atol=1e-2)(actual, expected)


if __name__ == '__main__':
  absltest.main()
