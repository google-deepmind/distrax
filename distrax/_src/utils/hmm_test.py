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
"""Tests for hmm.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import mvn_diag
from distrax._src.distributions import normal
from distrax._src.utils import hmm
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _make_models(init_logits, trans_logits, obs_dist_name, obs_params, length):
  """Build distrax HMM and equivalent TFP HMM."""
  obs_dist = {
      "categorical": categorical.Categorical,
      "normal": normal.Normal,
      "mvn_diag": mvn_diag.MultivariateNormalDiag,
  }[obs_dist_name](*obs_params)

  dx_model = hmm.HMM(
      init_dist=categorical.Categorical(init_logits),
      trans_dist=categorical.Categorical(trans_logits),
      obs_dist=obs_dist,
  )

  tfp_obs_dist = {
      "categorical": tfd.Categorical,
      "normal": tfd.Normal,
      "mvn_diag": tfd.MultivariateNormalDiag,
  }[obs_dist_name](*obs_params)

  tfp_model = tfd.HiddenMarkovModel(
      initial_distribution=tfd.Categorical(init_logits),
      transition_distribution=tfd.Categorical(trans_logits),
      observation_distribution=tfp_obs_dist,
      num_steps=length,
  )

  return dx_model, tfp_model


class Function:
  """Overrides lambda __repr__ to "fn" to stabilize test naming across cores."""

  def __init__(self, fn):
    self._fn = fn

  def __call__(self, *args, **kwargs):
    return self._fn(*args, **kwargs)

  def __repr__(self):
    return "fn"


def _test_cases(test_fn):
  return parameterized.product(
      length=(1, 17),
      num_states=(2, 23),
      obs_dist_name_and_params_fn=(
          ("categorical", Function(lambda n: (  # pylint: disable=g-long-lambda
              jax.random.normal(jax.random.PRNGKey(0), (n, 7)),))),
          ("normal", Function(lambda n: (  # pylint: disable=g-long-lambda
              jax.random.normal(jax.random.PRNGKey(0), (n,)),
              jax.random.normal(jax.random.PRNGKey(1), (n,))**2))),
          ("mvn_diag", Function(lambda n: (  # pylint: disable=g-long-lambda
              jax.random.normal(jax.random.PRNGKey(0), (n, 7)),
              jax.random.normal(jax.random.PRNGKey(1), (n, 7))**2))),
      ),
  )(test_fn)


class HMMTest(parameterized.TestCase):

  @chex.all_variants(without_device=False)
  @_test_cases
  def test_sample(self, length, num_states, obs_dist_name_and_params_fn):
    name, params_fn = obs_dist_name_and_params_fn

    logits = jax.random.normal(jax.random.PRNGKey(0), (num_states,))
    matrix = jax.random.normal(jax.random.PRNGKey(1), (num_states, num_states))

    model, tfp_model = _make_models(
        init_logits=logits, trans_logits=matrix, obs_dist_name=name,
        obs_params=params_fn(num_states), length=length)

    states, obs = self.variant(functools.partial(model.sample, seq_len=length))(
        seed=jax.random.PRNGKey(0))
    tfp_obs = tfp_model.sample(seed=jax.random.PRNGKey(0))

    with self.subTest("states"):
      chex.assert_type(states, jnp.int32)
      chex.assert_shape(states, (length,))

    with self.subTest("observations"):
      chex.assert_type(obs, model.obs_dist.dtype)
      chex.assert_shape(obs, (length, *model.obs_dist.event_shape))

    with self.subTest("matches TFP"):
      chex.assert_equal_shape([obs, tfp_obs])

  @chex.all_variants(without_device=False)
  @_test_cases
  def test_forward_backward(
      self, length, num_states, obs_dist_name_and_params_fn):
    name, params_fn = obs_dist_name_and_params_fn

    logits = jax.random.normal(jax.random.PRNGKey(0), (num_states,))
    matrix = jax.random.normal(jax.random.PRNGKey(1), (num_states, num_states))

    model, tfp_model = _make_models(
        init_logits=logits, trans_logits=matrix, obs_dist_name=name,
        obs_params=params_fn(num_states), length=length)

    _, observations = model.sample(seed=jax.random.PRNGKey(42), seq_len=length)
    alphas, betas, marginals, log_prob = self.variant(model.forward_backward)(
        observations)
    tfp_marginal_logits = tfp_model.posterior_marginals(observations).logits
    tfp_marginals = jax.nn.softmax(tfp_marginal_logits)

    with self.subTest("alphas"):
      chex.assert_type(alphas, jnp.float32)
      chex.assert_shape(alphas, (length, num_states))

    with self.subTest("betas"):
      chex.assert_type(betas, jnp.float32)
      chex.assert_shape(betas, (length, num_states))

    with self.subTest("marginals"):
      chex.assert_type(marginals, jnp.float32)
      chex.assert_shape(marginals, (length, num_states))

    with self.subTest("log_prob"):
      chex.assert_type(log_prob, jnp.float32)
      chex.assert_shape(log_prob, ())

    with self.subTest("matches TFP"):
      np.testing.assert_array_almost_equal(marginals, tfp_marginals, decimal=4)

  @chex.all_variants(without_device=False)
  @_test_cases
  def test_viterbi(self, length, num_states, obs_dist_name_and_params_fn):
    name, params_fn = obs_dist_name_and_params_fn

    logits = jax.random.normal(jax.random.PRNGKey(0), (num_states,))
    matrix = jax.random.normal(jax.random.PRNGKey(1), (num_states, num_states))

    model, tfp_model = _make_models(
        init_logits=logits, trans_logits=matrix, obs_dist_name=name,
        obs_params=params_fn(num_states), length=length)

    _, observations = model.sample(seed=jax.random.PRNGKey(42), seq_len=length)
    most_likely_states = self.variant(model.viterbi)(observations)
    tfp_mode = tfp_model.posterior_mode(observations)

    with self.subTest("shape"):
      chex.assert_shape(most_likely_states, (length,))

    with self.subTest("matches TFP"):
      np.testing.assert_array_equal(most_likely_states, tfp_mode)

  @chex.all_variants(without_device=False)
  def test_viterbi_matches_specific_example(self):
    loc = jnp.array([0.0, 1.0, 2.0, 3.0])
    scale = jnp.array(0.25)
    initial = jnp.array([0.25, 0.25, 0.25, 0.25])
    trans = jnp.array([[0.9, 0.1, 0.0, 0.0],
                       [0.1, 0.8, 0.1, 0.0],
                       [0.0, 0.1, 0.8, 0.1],
                       [0.0, 0.0, 0.1, 0.9]])

    observations = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 2.9, 2.8, 2.7, 2.6])

    model = hmm.HMM(
        init_dist=categorical.Categorical(probs=initial),
        trans_dist=categorical.Categorical(probs=trans),
        obs_dist=normal.Normal(loc, scale))

    inferred_states = self.variant(model.viterbi)(observations)
    expected_states = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    np.testing.assert_array_equal(inferred_states, expected_states)


if __name__ == "__main__":
  absltest.main()
