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
"""Quantized scoring paths must retain the configured numerical gap."""

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax
import jax.numpy as jnp
import numpy as np


class QuantizedEpsTest(parameterized.TestCase):

  @parameterized.product(
      base=[distrax.Normal, distrax.Logistic],
      eps=[None, 0.01, 0.1],
      bounded=[False, True],
  )
  def test_sample_and_log_prob_matches_separate_calls(self, base, eps, bounded):
    distribution = distrax.Quantized(
        base(jnp.zeros(2), jnp.array([1.0, 10000.0])),
        low=-100000.0 if bounded else None,
        high=100000.0 if bounded else None,
        eps=eps,
    )
    seed = jax.random.PRNGKey(19)
    for operation in (
        lambda d: d.sample_and_log_prob(seed=seed, sample_shape=(4, 3)),
        jax.jit(
            lambda d: d.sample_and_log_prob(seed=seed, sample_shape=(4, 3))
        ),
    ):
      samples, scores = operation(distribution)
      np.testing.assert_array_equal(
          samples, distribution.sample(seed=seed, sample_shape=(4, 3))
      )
      np.testing.assert_allclose(
          scores, distribution.log_prob(samples), rtol=2e-5, atol=1e-5
      )

  @parameterized.product(eps=[0.01, 0.1], use_jit=[False, True])
  def test_combined_scores_and_parameter_gradients_match_the_separate_path(
      self, eps, use_jit
  ):
    def objective(loc, combined):
      distribution = distrax.Quantized(distrax.Normal(loc, 10000.0), eps=eps)
      seed = jax.random.PRNGKey(5)
      if combined:
        _, scores = distribution.sample_and_log_prob(seed=seed, sample_shape=8)
      else:
        samples = distribution.sample(seed=seed, sample_shape=8)
        scores = distribution.log_prob(samples)
      return jnp.sum(scores)

    actual = jax.value_and_grad(lambda loc: objective(loc, True))
    expected = jax.value_and_grad(lambda loc: objective(loc, False))
    if use_jit:
      actual, expected = jax.jit(actual), jax.jit(expected)
    for a, b in zip(actual(jnp.array(0.0)), expected(jnp.array(0.0))):
      self.assertTrue(bool(jnp.isfinite(a)))
      np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-8)

  @parameterized.product(
      eps_kind=['none', 'scalar', 'batch'], index=[0, (slice(None), 1), (1, 2)]
  )
  def test_batch_slicing_preserves_broadcast_gap(self, eps_kind, index):
    eps = {'none': None, 'scalar': 0.1, 'batch': jnp.array([[0.01], [0.1]])}[
        eps_kind
    ]
    distribution = distrax.Quantized(
        distrax.Normal(jnp.zeros((2, 3)), 10000.0), low=-5.0, high=5.0, eps=eps
    )
    sliced = distribution[index]
    for value in [-5.0, 0.0, 5.0]:
      np.testing.assert_allclose(
          sliced.log_prob(jnp.array(value)),
          distribution.log_prob(jnp.array(value))[index],
      )
    values, scores = jax.jit(
        lambda d: d.sample_and_log_prob(
            seed=jax.random.PRNGKey(1), sample_shape=4
        )
    )(sliced)
    np.testing.assert_allclose(scores, sliced.log_prob(values))


if __name__ == '__main__':
  absltest.main()
