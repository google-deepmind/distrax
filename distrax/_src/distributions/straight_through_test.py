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
"""Tests for `straight_through.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.distributions import one_hot_categorical
from distrax._src.distributions import straight_through
from distrax._src.utils import equivalence
from distrax._src.utils import math
import jax
import jax.numpy as jnp
import numpy as np


class StraightThroughTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(
        straight_through.straight_through_wrapper(
            one_hot_categorical.OneHotCategorical))

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d logits, no shape', {'logits': [0.0, 1.0, -0.5]}, ()),
      ('1d probs, no shape', {'probs': [0.2, 0.5, 0.3]}, ()),
      ('1d logits, int shape', {'logits': [0.0, 1.0, -0.5]}, 1),
      ('1d probs, int shape', {'probs': [0.2, 0.5, 0.3]}, 1),
      ('1d logits, 1-tuple shape', {'logits': [0.0, 1.0, -0.5]}, (1,)),
      ('1d probs, 1-tuple shape', {'probs': [0.2, 0.5, 0.3]}, (1,)),
      ('1d logits, 2-tuple shape', {'logits': [0.0, 50., -0.5]}, (5, 4)),
      ('1d probs, 2-tuple shape', {'probs': [0.01, 0.99, 0.]}, (5, 4)),
      ('2d logits, no shape', {'logits': [[0.0, 1.0, -0.5],
                                          [-0.1, 0.3, 0.0]]}, ()),
      ('2d probs, no shape', {'probs': [[0.1, 0.4, 0.5],
                                        [0.5, 0.25, 0.25]]}, ()),
      ('2d logits, int shape', {'logits': [[0.0, 50.0, -0.5],
                                           [-0.1, -0.3, 0.2]]}, 4),
      ('2d probs, int shape', {'probs': [[0.005, 0.005, 0.99],
                                         [0.99, 0., 0.01]]}, 4),
      ('2d logits, 1-tuple shape', {'logits': [[0.0, 1.0, -0.5],
                                               [-0.1, 0.3, 200.0]]}, (5,)),
      ('2d probs, 1-tuple shape', {'probs': [[0., 0.01, 0.99],
                                             [0., 0.99, 0.01]]}, (5,)),
      ('2d logits, 2-tuple shape', {'logits': [[0.0, 1.0, -0.5],
                                               [-0.1, 0.3, 1000.0]]}, (5, 4)),
      ('2d probs, 2-tuple shape', {'probs': [[0.01, 0.99, 0.],
                                             [0.99, 0., 0.01]]}, (5, 4)),
  )
  def test_sample(self, dist_params, sample_shape):

    def loss(dist_params, dist_cls, sample_shape):
      """Loss on sample, used both for distrax and TFP."""

      # Sample.
      dist = dist_cls(**dist_params)
      sample_fn = dist.sample

      def sample_fn_wrapper(seed, sample_shape):
        """To test with pmap that requires positional arguments."""
        return sample_fn(seed=seed, sample_shape=sample_shape)

      if hasattr(self, 'variant'):
        sample_fn_wrapper = self.variant(static_argnums=(1,))(sample_fn_wrapper)
      sample = sample_fn_wrapper(self.key, sample_shape)
      return jnp.sum((sample)**2).astype(jnp.float32), sample

    # TFP softmax gradient.
    def straight_through_tfp_loss(dist_params, dist_cls, sample_shape):
      """Loss on a straight-through gradient of the tfp sample."""
      # Distrax normalises the distribution parameters. We want to make sure
      # that they are normalised for tfp too, or the gradient might differ.
      try:
        dist_params['logits'] = math.normalize(logits=dist_params['logits'])
      except KeyError:
        dist_params['probs'] = math.normalize(probs=dist_params['probs'])

      # Sample.
      dist = dist_cls(**dist_params)
      sample_fn = dist.sample

      def sample_fn_wrapper(seed, sample_shape):
        """To test with pmap that requires positional arguments."""
        return sample_fn(seed=seed, sample_shape=sample_shape)

      if hasattr(self, 'variant'):
        sample_fn_wrapper = self.variant(static_argnums=(1,))(sample_fn_wrapper)
      sample = sample_fn_wrapper(self.key, sample_shape)

      # Straight-through gradient.
      def _pad(probs, shape):
        if isinstance(shape, int):
          return probs
        while len(probs.shape) < len(shape):
          probs = probs[None]
        return probs
      probs = dist.probs_parameter()
      padded_probs = _pad(probs, sample_shape)
      sample += padded_probs - jax.lax.stop_gradient(padded_probs)

      return jnp.sum((sample)**2).astype(jnp.float32), sample

    # Straight-through gradient and sample.
    sample_grad, sample = jax.grad(loss, has_aux=True)(dist_params,
                                                       self.distrax_cls,
                                                       sample_shape)
    # TFP gradient (zero) and sample.
    tfp_sample_grad, tfp_sample = jax.grad(loss, has_aux=True)(dist_params,
                                                               self.tfp_cls,
                                                               sample_shape)
    # TFP straight-through gradient and sample.
    tfp_st_sample_grad, tfp_st_sample = jax.grad(straight_through_tfp_loss,
                                                 has_aux=True)(dist_params,
                                                               self.tfp_cls,
                                                               sample_shape)

    # TEST: the samples have the same size, and the straight-through gradient
    # doesn't affect the tfp sample.
    chex.assert_equal_shape((sample, tfp_sample))
    self.assertion_fn(rtol=2e-3)(tfp_sample, tfp_st_sample)
    # TEST: the TFP gradient is zero.
    assert (jnp.asarray(*tfp_sample_grad.values()) == 0).all()
    # TEST: the TFP straight-through gradient is non zero.
    assert (jnp.asarray(*tfp_st_sample_grad.values()) != 0).any()
    # Test that the TFP straight-through gradient is equal to the one from
    # distrax when the samples from distrax and tfp are the same (due to
    # stochasticity the samples can differ - we are using skewed distributions
    # on purpose in the parametrization of the test to make sure that the
    # samples match most of the time).
    sample_grad_v = jnp.stack(jnp.array(*sample_grad.values()))
    tfp_st_sample_grad_v = jnp.stack(jnp.array(*tfp_st_sample_grad.values()))
    if np.all(sample == tfp_st_sample):
      self.assertion_fn(rtol=2e-3)(sample_grad_v, tfp_st_sample_grad_v)


if __name__ == '__main__':
  absltest.main()
