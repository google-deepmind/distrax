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
"""Variational Autoencoder example on binarized MNIST dataset."""

from typing import Any, Iterator, Mapping, NamedTuple, Sequence, Tuple

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


flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
FLAGS = flags.FLAGS


OptState = Any
PRNGKey = jnp.ndarray
Batch = Mapping[str, np.ndarray]

MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)


def load_dataset(split: str, batch_size: int) -> Iterator[Batch]:
  ds = tfds.load("binarized_mnist", split=split, shuffle_files=True)
  ds = ds.shuffle(buffer_size=10 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, hidden_size: int = 512, latent_size: int = 10):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = hk.Flatten()(x)
    x = hk.Linear(self._hidden_size)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(self._latent_size)(x)
    log_stddev = hk.Linear(self._latent_size)(x)
    stddev = jnp.exp(log_stddev)

    return mean, stddev


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(
      self,
      hidden_size: int = 512,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    z = hk.Linear(self._hidden_size)(z)
    z = jax.nn.relu(z)

    logits = hk.Linear(np.prod(self._output_shape))(z)
    logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits


class VAEOutput(NamedTuple):
  variational_distrib: distrax.Distribution
  likelihood_distrib: distrax.Distribution
  image: jnp.ndarray


class VAE(hk.Module):
  """Main VAE model class, uses Encoder & Decoder under the hood."""

  def __init__(
      self,
      latent_size: int = 10,
      hidden_size: int = 512,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._latent_size = latent_size
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, x: jnp.ndarray) -> VAEOutput:
    x = x.astype(jnp.float32)

    # q(z|x) = N(mean(x), covariance(x))
    mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
    variational_distrib = distrax.MultivariateNormalDiag(
        loc=mean, scale_diag=stddev)
    z = variational_distrib.sample(seed=hk.next_rng_key())

    # p(x|z) = \Prod Bernoulli(logits(z))
    logits = Decoder(self._hidden_size, self._output_shape)(z)
    likelihood_distrib = distrax.Independent(
        distrax.Bernoulli(logits=logits),
        reinterpreted_batch_ndims=len(self._output_shape))  # 3 non-batch dims

    # Generate images from the likelihood
    image = likelihood_distrib.sample(seed=hk.next_rng_key())

    return VAEOutput(variational_distrib, likelihood_distrib, image)


def main(_):
  latent_size = 10

  model = hk.transform(
      lambda x: VAE(latent_size)(x),  # pylint: disable=unnecessary-lambda
      apply_rng=True)
  optimizer = optax.adam(FLAGS.learning_rate)

  @jax.jit
  def loss_fn(params: hk.Params, rng_key: PRNGKey, batch: Batch) -> jnp.ndarray:
    """Loss = -ELBO, where ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))."""

    outputs: VAEOutput = model.apply(params, rng_key, batch["image"])

    # p(z) = N(0, I)
    prior_z = distrax.MultivariateNormalDiag(
        loc=jnp.zeros((latent_size,)),
        scale_diag=jnp.ones((latent_size,)))

    log_likelihood = outputs.likelihood_distrib.log_prob(batch["image"])
    kl = outputs.variational_distrib.kl_divergence(prior_z)
    elbo = log_likelihood - kl

    return -jnp.mean(elbo)

  @jax.jit
  def update(
      params: hk.Params,
      rng_key: PRNGKey,
      opt_state: OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  rng_seq = hk.PRNGSequence(42)
  params = model.init(next(rng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)))
  opt_state = optimizer.init(params)

  train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  for step in range(FLAGS.training_steps):
    params, opt_state = update(params, next(rng_seq), opt_state, next(train_ds))

    if step % FLAGS.eval_frequency == 0:
      val_loss = loss_fn(params, next(rng_seq), next(valid_ds))
      logging.info("STEP: %5d; Validation ELBO: %.3f", step, -val_loss)


if __name__ == "__main__":
  app.run(main)
