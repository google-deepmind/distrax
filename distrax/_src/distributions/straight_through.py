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
"""Straight-through gradient sampling distribution."""
from distrax._src.distributions import categorical
from distrax._src.distributions import distribution
import jax


def straight_through_wrapper(  # pylint: disable=invalid-name
    Distribution,
    ) -> distribution.DistributionLike:
  """Wrap a distribution to use straight-through gradient for samples."""

  def sample(self, seed, sample_shape=()):  # pylint: disable=g-doc-args
    """Sampling with straight through biased gradient estimator.

    Sample a value from the distribution, but backpropagate through the
    underlying probability to compute the gradient.

    References:
      [1] Yoshua Bengio, Nicholas Léonard, Aaron Courville, Estimating or
      Propagating Gradients Through Stochastic Neurons for Conditional
      Computation, https://arxiv.org/abs/1308.3432

    Args:
      seed: a random seed.
      sample_shape: the shape of the required sample.

    Returns:
      A sample with straight-through gradient.
    """
    # pylint: disable=protected-access
    obj = Distribution(probs=self._probs, logits=self._logits)
    assert isinstance(obj, categorical.Categorical)
    sample = obj.sample(seed=seed, sample_shape=sample_shape)
    probs = obj.probs
    padded_probs = _pad(probs, sample.shape)

    # Keep sample unchanged, but add gradient through probs.
    sample += padded_probs - jax.lax.stop_gradient(padded_probs)
    return sample

  def _pad(probs, shape):
    """Grow probs to have the same number of dimensions as shape."""
    while len(probs.shape) < len(shape):
      probs = probs[None]
    return probs

  parent_name = Distribution.__name__
  # Return a new object, overriding sample.
  return type('StraighThrough' + parent_name, (Distribution,),
              {'sample': sample})
