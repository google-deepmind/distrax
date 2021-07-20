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
"""A simple example of a flow model trained on MNIST."""

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds


flags.DEFINE_integer("flow_num_layers", 8,
                     "Number of layers to use in the flow.")
flags.DEFINE_integer("mlp_num_layers", 2,
                     "Number of layers to use in the MLP conditioner.")
flags.DEFINE_integer("hidden_size", 500, "Hidden size of the MLP conditioner.")
flags.DEFINE_integer("num_bins", 4,
                     "Number of bins to use in the rational-quadratic spline.")
flags.DEFINE_integer("batch_size", 128,
                     "Batch size for training and evaluation.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
FLAGS = flags.FLAGS

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any

MNIST_IMAGE_SHAPE = (28, 28, 1)


def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Sequential:
  """Creates an MLP conditioner for each layer of the flow."""
  return hk.Sequential([
      hk.Flatten(preserve_dims=-len(event_shape)),
      hk.nets.MLP(hidden_sizes, activate_final=True),
      # We initialize this linear layer to zero so that the flow is initialized
      # to the identity function.
      hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros),
      hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
  ])


def make_flow_model(event_shape: Sequence[int],
                    num_layers: int,
                    hidden_sizes: Sequence[int],
                    num_bins: int) -> distrax.Transformed:
  """Creates the flow model."""
  # Alternating binary mask.
  mask = jnp.arange(0, np.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=0., range_max=1.)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.
  num_bijector_params = 3 * num_bins + 1

  layers = []
  for _ in range(num_layers):
    layer = distrax.MaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=make_conditioner(event_shape, hidden_sizes,
                                     num_bijector_params))
    layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # We invert the flow so that the `forward` method is called with `log_prob`.
  flow = distrax.Inverse(distrax.Chain(layers))
  base_distribution = distrax.Independent(
      distrax.Uniform(
          low=jnp.zeros(event_shape),
          high=jnp.ones(event_shape)),
      reinterpreted_batch_ndims=len(event_shape))

  return distrax.Transformed(base_distribution, flow)


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
  ds = tfds.load("mnist", split=split, shuffle_files=True)
  ds = ds.shuffle(buffer_size=10 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
  data = batch["image"].astype(np.float32)
  if prng_key is not None:
    # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
    data += jax.random.uniform(prng_key, data.shape)
  return data / 256.  # Normalize pixel values from [0, 256) to [0, 1).


@hk.without_apply_rng
@hk.transform
def log_prob(data: Array) -> Array:
  model = make_flow_model(
      event_shape=data.shape[1:],
      num_layers=FLAGS.flow_num_layers,
      hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
      num_bins=FLAGS.num_bins)
  return model.log_prob(data)


def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:
  data = prepare_data(batch, prng_key)
  # Loss is average negative log likelihood.
  loss = -jnp.mean(log_prob.apply(params, data))
  return loss


@jax.jit
def eval_fn(params: hk.Params, batch: Batch) -> Array:
  data = prepare_data(batch)  # We don't dequantize during evaluation.
  loss = -jnp.mean(log_prob.apply(params, data))
  return loss


def main(_):
  optimizer = optax.adam(FLAGS.learning_rate)

  @jax.jit
  def update(params: hk.Params,
             prng_key: PRNGKey,
             opt_state: OptState,
             batch: Batch) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  prng_seq = hk.PRNGSequence(42)
  params = log_prob.init(next(prng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)))
  opt_state = optimizer.init(params)

  train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  for step in range(FLAGS.training_steps):
    params, opt_state = update(params, next(prng_seq), opt_state,
                               next(train_ds))

    if step % FLAGS.eval_frequency == 0:
      val_loss = eval_fn(params, next(valid_ds))
      logging.info("STEP: %5d; Validation loss: %.3f", step, val_loss)


if __name__ == "__main__":
  app.run(main)
