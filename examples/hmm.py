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
"""Hidden Markov Model example detecting changepoints in the rate of a signal.

Adapted from https://github.com/probml/probml-notebooks/blob/main/notebooks/
                 hmm_poisson_changepoint_jax.ipynb
"""

import functools

from absl import app
from absl import flags
from absl import logging
import distrax
import jax
import jax.numpy as jnp
import optax
import scipy.stats
import tensorflow_probability as tfp


flags.DEFINE_list("true_rates", [40, 3, 20, 50],
                  "Sequence of Poisson rates for the data generating process.")

flags.DEFINE_list("true_durations", [10, 20, 5, 35],
                  "Sequence of durations for the data generating process. "
                  "Should be the same length as `true_rates`.")

flags.DEFINE_integer("fixed_num_states", 4,
                     "How many states to use for the fixed-count experiment.")

flags.DEFINE_list("state_sweep", [1, 2, 3, 4, 5, 6],
                  "Sweep of states to use for the multi-count experiment.")

flags.DEFINE_float("prior_change_prob", 0.05,
                   "Prior probability of state transition per unit time.")

flags.DEFINE_integer("n_steps", 201,
                     "Number of steps of gradient descent to fit the model.")

flags.DEFINE_integer("data_seed", 0, "Seed for the data generator.")

flags.DEFINE_integer("model_seed", 1, "Seed for the parameter generator.")

FLAGS = flags.FLAGS


def generate_data(true_rates, true_durations, random_state):
  """Generates data from a Poisson process with changing rates over time."""
  return jnp.concatenate([
      scipy.stats.poisson(rate).rvs(num_steps, random_state=random_state)
      for (rate, num_steps) in zip(true_rates, true_durations)
  ]).astype(jnp.float32)


def build_latent_state(num_states, max_num_states, daily_change_prob):
  """"Build an initial state probability vector and state transition matrix."""
  # Give probability 0 to states outside of the current model.
  def prob(s):
    return jnp.where(s < num_states + 1, 1/num_states, 0.)

  states = jnp.arange(1, max_num_states+1)
  initial_state_probs = jax.vmap(prob)(states)

  # Build a transition matrix that transitions only within the current
  # `num_states` states.
  def transition_prob(i, s):
    return jnp.where((s <= num_states) & (i <= num_states) & (1 < num_states),
                     jnp.where(s == i, 1 - daily_change_prob,
                               daily_change_prob / (num_states - 1)),
                     jnp.where(s == i, 1, 0))

  transition_probs = jax.vmap(
      transition_prob, in_axes=(None, 0))(states, states)

  return initial_state_probs, transition_probs


def make_hmm(log_rates, transition_probs, initial_state_probs):
  """Make a Hidden Markov Model with Poisson observation distribution."""
  return distrax.HMM(
      obs_dist=tfp.substrates.jax.distributions.Poisson(log_rate=log_rates),
      trans_dist=distrax.Categorical(probs=transition_probs),
      init_dist=distrax.Categorical(probs=initial_state_probs))


def get_durations(data):
  durations = []
  previous_value = None
  for value in data:
    if value != previous_value:
      durations.append(1)
      previous_value = value
    else:
      durations[-1] += 1
  return durations


def get_changed_rates(data):
  values = []
  for value in data:
    if not values or value != values[-1]:
      values.append(value)
  return values


def main(_):
  #--------------------------------------------------
  #-------------- Generate the data -----------------
  #--------------------------------------------------
  observed_counts = generate_data(FLAGS.true_rates,
                                  FLAGS.true_durations,
                                  FLAGS.data_seed)

  #-----------------------------------------------------------------------
  #-------------- Run a model with fixed number of states ----------------
  #-----------------------------------------------------------------------
  initial_state_probs, transition_probs = build_latent_state(
      FLAGS.fixed_num_states, FLAGS.fixed_num_states, FLAGS.prior_change_prob)

  logging.info("--------- Fixed number of states ---------")
  logging.info("Initial state probs: %s", initial_state_probs)
  logging.info("Transition matrix:\n%s", transition_probs)

  rng_key = jax.random.PRNGKey(FLAGS.model_seed)
  rng_key, rng_normal = jax.random.split(rng_key)

  # Define a variable to represent the unknown log-rates.
  trainable_log_rates = (
      jnp.log(jnp.mean(observed_counts))
      + jax.random.normal(rng_normal, (FLAGS.fixed_num_states,)))
  hmm = make_hmm(trainable_log_rates, transition_probs, initial_state_probs)

  optimizer = optax.adam(1e-1)

  # Define loss and update functions for doing gradient descent.
  def loss_fn(trainable_log_rates, transition_probs, initial_state_probs):
    """Computes the loss for the model given the log-rates."""
    hmm = make_hmm(trainable_log_rates, transition_probs, initial_state_probs)
    rate_prior = distrax.LogStddevNormal(5, 5)
    return -(jnp.sum(rate_prior.log_prob(jnp.exp(trainable_log_rates)))
             + hmm.forward(observed_counts)[0])

  def update(opt_state, params, transition_probs, initial_state_probs):
    """Computes the gradient and updates the parameters of the model."""
    loss, grads = jax.value_and_grad(loss_fn)(
        params, transition_probs, initial_state_probs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss

  @functools.partial(jax.jit, static_argnums=3)
  def fit(trainable_log_rates, transition_probs, initial_state_probs, n_steps):
    """Does n_steps of gradient descent on the model."""
    opt_state = optimizer.init(trainable_log_rates)
    def train_step(opt_state_params, _):
      opt_state, params = opt_state_params
      opt_state, params, loss = update(
          opt_state, params, transition_probs, initial_state_probs)
      return (opt_state, params), loss

    steps = jnp.arange(n_steps)
    (opt_state, trainable_log_rates), losses = jax.lax.scan(
        train_step, (opt_state, trainable_log_rates), steps)

    return trainable_log_rates, losses

  # Do gradient descent to fit the model.
  params, losses = fit(
      trainable_log_rates, transition_probs, initial_state_probs, FLAGS.n_steps)
  rates = jnp.exp(params)
  hmm = make_hmm(params, transition_probs, initial_state_probs)

  logging.info("Initial loss: %s", losses[0])
  logging.info("Final loss: %s", losses[-1])
  logging.info("Inferred rates: %s", rates)
  logging.info("True rates: %s", FLAGS.true_rates)

  _, _, posterior_probs, _ = hmm.forward_backward(observed_counts)

  # Max marginals
  most_probable_states = jnp.argmax(posterior_probs, axis=-1)
  most_probable_rates = rates[most_probable_states]
  logging.info("Inferred rates between change points (Max marginals): %s",
               get_changed_rates(most_probable_rates))
  logging.info("Inferred durations between change points (max marginals): %s",
               get_durations(most_probable_states))

  # Max probability trajectory (Viterbi)
  most_probable_states = hmm.viterbi(observed_counts)
  most_probable_rates = rates[most_probable_states]
  logging.info("Inferred rates between change points (Viterbi): %s",
               get_changed_rates(most_probable_rates))
  logging.info("Inferred durations between change points (Viterbi): %s",
               get_durations(most_probable_states))

  #----------------------------------------------------------------------------
  #-------- Run a sweep over models with different numbers of states ----------
  #----------------------------------------------------------------------------

  states = jnp.array(FLAGS.state_sweep)

  # For each candidate model, build initial state prior and transition matrix
  batch_initial_state_probs, batch_transition_probs = jax.vmap(
      build_latent_state, in_axes=(0, None, None))(
          states, max(FLAGS.state_sweep), FLAGS.prior_change_prob)

  logging.info("----- Sweeping over models with different state counts -----")
  logging.info("Shape of initial_state_probs: %s",
               batch_initial_state_probs.shape)
  logging.info("Shape of transition_probs: %s", batch_transition_probs.shape)
  logging.info("Example initial_state_probs for num_states==%s: %s",
               FLAGS.state_sweep[2], batch_initial_state_probs[2, :])
  logging.info("Example transition_probs for num_states==%s:\n%s",
               FLAGS.state_sweep[2], batch_transition_probs[2, :])

  rng_key, rng_normal = jax.random.split(rng_key)

  # Define a variable to represent the unknown log-rates.
  trainable_log_rates = (
      jnp.log(jnp.mean(observed_counts))
      + jax.random.normal(rng_normal, (max(FLAGS.state_sweep),)))

  # Fit the model with gradient descent.
  params, losses = jax.vmap(fit, in_axes=(None, 0, 0, None))(
      trainable_log_rates, batch_transition_probs, batch_initial_state_probs,
      FLAGS.n_steps)
  rates = jnp.exp(params)
  logging.info("Final loss for each model: %s", losses[:, -1])

  for i, learned_model_rates in enumerate(rates):
    logging.info("Rates for %s-state model: %s",
                 FLAGS.state_sweep[i], learned_model_rates[:i+1])

  def posterior_marginals(
      trainable_log_rates, initial_state_probs, transition_probs):
    hmm = make_hmm(trainable_log_rates, transition_probs, initial_state_probs)
    _, _, marginals, _ = hmm.forward_backward(observed_counts)
    return marginals

  posterior_probs = jax.vmap(posterior_marginals, in_axes=(0, 0, 0))(
      params, batch_initial_state_probs, batch_transition_probs)
  most_probable_states = jnp.argmax(posterior_probs, axis=-1)

  for i, learned_model_rates in enumerate(rates):
    logging.info("%s-state model:", FLAGS.state_sweep[i])
    logging.info(
        "Inferred rates between change points: %s",
        get_changed_rates(learned_model_rates[most_probable_states[i]]))
    logging.info(
        "Inferred durations between change points: %s",
        get_durations(most_probable_states[i]))


if __name__ == "__main__":
  app.run(main)
