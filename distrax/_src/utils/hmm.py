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
"""Hidden Markov Model implementation."""

from typing import Optional, Tuple

import chex
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
from distrax._src.utils import jittable
import jax
import jax.numpy as jnp


def _normalize(u: chex.Array,
               axis: int = 0,
               eps: float = 1e-15) -> Tuple[chex.Array, chex.Array]:
  """Normalizes the values within the axis in a way that they sum up to 1.

  Args:
    u: Input array to normalize.
    axis: Axis over which to normalize.
    eps: Minimum value threshold for numerical stability.

  Returns:
    Tuple of the normalized values, and the normalizing denominator.
  """
  u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
  c = u.sum(axis=axis)
  c = jnp.where(c == 0, 1, c)
  return u / c, c


class HMM(jittable.Jittable):
  """Hidden Markov Model class."""

  def __init__(self,
               init_dist: categorical.CategoricalLike,
               trans_dist: categorical.CategoricalLike,
               obs_dist: distribution.DistributionLike):
    """Constructs an N-state Hidden Markov Model from component distributions.

    Args:
      init_dist: Integer-valued categorical distribution with parameters of
        shape (N,), representing the distribution over initial latent states.
      trans_dist: Integer-valued categorical distribution with parameters of
        shape (N, N), representing the transition probability matrix between
        latent states.
      obs_dist: Any observation distribution with batch shape (N,), representing
        `p(observation|latent state)`.
    """
    self._init_dist = conversion.as_distribution(init_dist)
    self._trans_dist = conversion.as_distribution(trans_dist)
    self._obs_dist = conversion.as_distribution(obs_dist)
    self._n_states = self._init_dist.num_categories

    if not jnp.issubdtype(self._init_dist.dtype, jnp.integer):
      raise TypeError(
          f'init_dist must be categorical-like with integer dtype, but its '
          f'dtype is {self._init_dist.dtype}.')

    if not jnp.issubdtype(self._trans_dist.dtype, jnp.integer):
      raise TypeError(
          f'trans_dist must be categorical-like with integer dtype, but its '
          f'dtype is {self._trans_dist.dtype}.')

    if self._init_dist.batch_shape:
      raise ValueError(
          f'init_dist must be unbatched, but it has a batch shape of '
          f'{self._init_dist.batch_shape}.')

    if self._obs_dist.batch_shape != (self._n_states,):
      raise ValueError(
          f'obs_dist should have batch shape of ({self._n_states},) equal to '
          f'the number of latent states in the model, but its batch shape is '
          f'{self._obs_dist.batch_shape}.')

    if self._trans_dist.batch_shape != (self._n_states,):
      raise ValueError(
          f'trans_dist should have batch shape of ({self._n_states},) equal to '
          f'the number of latent states in the model, but its batch shape is '
          f'{self._trans_dist.batch_shape}.')

    if self._trans_dist.num_categories != self._n_states:
      raise ValueError(
          f'trans_dist should have `num_categories` of {self._n_states} equal '
          f'to the number of latent states in the model, but it has '
          f'`num_categories` of {self._trans_dist.num_categories}.')

  @property
  def init_dist(self) -> categorical.CategoricalLike:
    return self._init_dist

  @property
  def trans_dist(self) -> categorical.CategoricalLike:
    return self._trans_dist

  @property
  def obs_dist(self) -> distribution.DistributionLike:
    return self._obs_dist

  def sample(self,
             *,
             seed: chex.PRNGKey,
             seq_len: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Sample from this HMM.

    Samples an observation of given length according to this
    Hidden Markov Model and gives the sequence of the hidden states
    as well as the observation.

    Args:
      seed: Random key of shape (2,) and dtype uint32.
      seq_len: The length of the observation sequence.

    Returns:
      Tuple of hidden state sequence, and observation sequence.
    """
    rng_key, rng_init = jax.random.split(seed)
    initial_state = self._init_dist.sample(seed=rng_init)

    def draw_state(prev_state, key):
      state = self._trans_dist.sample(seed=key)[prev_state]
      return state, state

    rng_state, rng_obs = jax.random.split(rng_key)
    keys = jax.random.split(rng_state, seq_len - 1)
    _, states = jax.lax.scan(draw_state, initial_state, keys)
    states = jnp.append(initial_state, states)

    def draw_obs(state, key):
      return self._obs_dist.sample(seed=key)[state]

    keys = jax.random.split(rng_obs, seq_len)
    obs_seq = jax.vmap(draw_obs, in_axes=(0, 0))(states, keys)

    return states, obs_seq

  def forward(self,
              obs_seq: chex.Array,
              length: Optional[chex.Array] = None) -> Tuple[float, chex.Array]:
    """Calculates a belief state.

    Args:
      obs_seq: Observation sequence.
      length: The valid length of the observation sequence, used to truncate the
        computation for batches of varying length. If set to None, the entire
        sequence is used.

    Returns:
      Tuple of `log(p(x_{1:T}|model))` and the array of forward joint
        probabilities `p(z_t,x_{1:t})` for each sample `x_t`.
    """
    seq_len = len(obs_seq)

    if length is None:
      length = seq_len

    def scan_fn(carry, t):
      (alpha_prev, log_ll_prev) = carry
      alpha_n = jnp.where(
          t < length,
          (self._obs_dist.prob(obs_seq[t])
           * (alpha_prev[:, None] * self._trans_dist.probs).sum(axis=0)),
          jnp.zeros_like(alpha_prev))

      alpha_n, cn = _normalize(alpha_n)
      carry = (alpha_n, jnp.log(cn) + log_ll_prev)

      return carry, alpha_n

    # initial belief state
    alpha_0, c0 = _normalize(
        self._init_dist.probs * self._obs_dist.prob(obs_seq[0]))

    # setup scan loop
    init_state = (alpha_0, jnp.log(c0))
    ts = jnp.arange(1, seq_len)
    carry, alpha_hist = jax.lax.scan(scan_fn, init_state, ts)

    # post-process
    alpha_hist = jnp.vstack([alpha_0.reshape(1, self._n_states), alpha_hist])
    (_, log_ll) = carry
    return log_ll, alpha_hist

  def backward(self,
               obs_seq: chex.Array,
               length: Optional[chex.Array] = None) -> chex.Array:
    """Computes the backward probabilities.

    Args:
      obs_seq: Observation sequence.
      length: The valid length of the observation sequence, used to truncate the
        computation for batches of varying length. If set to None, the entire
        sequence is used.

    Returns:
      Array of backward joint probabilities `p(x_{t+1:T}|z_t)`.
    """
    seq_len = len(obs_seq)

    if length is None:
      length = seq_len

    beta_t = jnp.ones((self._n_states,))

    def scan_fn(beta_prev, t):
      beta_t = jnp.where(
          t > length,
          jnp.zeros_like(beta_prev),
          _normalize((beta_prev * self._obs_dist.prob(obs_seq[t-1])
                      * self._trans_dist.probs).sum(axis=1))[0])
      return beta_t, beta_t

    ts = jnp.arange(seq_len, 1, -1)
    _, beta_hist = jax.lax.scan(scan_fn, beta_t, ts)

    beta_hist = jnp.flip(
        jnp.vstack([beta_t.reshape(1, self._n_states), beta_hist]), axis=0)

    return beta_hist

  def forward_backward(
      self,
      obs_seq: chex.Array,
      length: Optional[chex.Array] = None,
  ) -> Tuple[chex.Array, chex.Array, chex.Array, float]:
    """HMM forward-backward algorithm.

    Computes, for each time step, the marginal conditional probability that the
    Hidden Markov Model was in each possible state given the observations that
    were made at each time step, i.e. P(z[i] | x[0], ..., x[num_steps - 1])
    for all i from 0 to num_steps - 1.

    Args:
      obs_seq: Observation sequence.
      length: The valid length of the observation sequence, used to truncate the
        computation for batches of varying length. If set to None, the entire
        sequence is used.

    Returns:
      Tuple of:
        * Forward joint probabilities `p(z_t,x_{1:t})`.
        * Backward joint probabilities `p(x_{t+1:T}|z_t)`.
        * Marginal conditional probability of the observations.
        * The log-likelihood log(p(x_{1:T}|model)).
    """
    seq_len = len(obs_seq)

    if length is None:
      length = seq_len

    def gamma_t(t):
      return alpha[t] * beta[t]

    ll, alpha = self.forward(obs_seq, length)

    beta = self.backward(obs_seq, length)

    ts = jnp.arange(seq_len)
    gamma = jax.vmap(gamma_t)(ts)
    gamma = jax.vmap(lambda x: _normalize(x)[0])(gamma)
    return alpha, beta, gamma, ll

  def viterbi(self, obs_seq: chex.Array) -> chex.Array:
    """Viterbi algorithm.

    Computes the most probable sequence of hidden states given the observations.

    Args:
      obs_seq: Observation sequence.

    Returns:
      The most probable sequence of hidden states.
    """

    trans_log_probs = jax.nn.log_softmax(self._trans_dist.logits)
    init_log_probs = jax.nn.log_softmax(self._init_dist.logits)

    first_log_prob = init_log_probs + self._obs_dist.log_prob(obs_seq[0])

    if len(obs_seq) == 1:
      return jnp.expand_dims(jnp.argmax(first_log_prob), axis=0)

    def viterbi_forward(prev_logp, obs):
      obs_logp = self._obs_dist.log_prob(obs)
      logp = prev_logp[..., None] + trans_log_probs + obs_logp[..., None, :]
      max_logp_given_successor = jnp.max(logp, axis=-2)
      most_likely_given_successor = jnp.argmax(logp, axis=-2)
      return max_logp_given_successor, most_likely_given_successor

    final_log_prob, most_likely_sources = jax.lax.scan(
        viterbi_forward, first_log_prob, obs_seq[1:])

    most_likely_initial_given_successor = jnp.argmax(
        trans_log_probs + first_log_prob, axis=-2)
    most_likely_sources = jnp.concatenate([
        jnp.expand_dims(most_likely_initial_given_successor, axis=0),
        most_likely_sources], axis=0)

    def viterbi_backward(state, most_likely_sources):
      state = jax.nn.one_hot(state, self._n_states)
      most_likely = jnp.sum(most_likely_sources * state).astype(jnp.int64)
      return most_likely, most_likely

    final_state = jnp.argmax(final_log_prob)
    _, most_likely_path = jax.lax.scan(
        viterbi_backward, final_state, most_likely_sources[1:], reverse=True)

    return jnp.append(most_likely_path, final_state)
