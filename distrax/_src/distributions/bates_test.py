# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
from distrax._src.distributions import bates
from distrax._src.utils import equivalence
import jax
# import jax.numpy as jnp  # Unused
import numpy as np


class BatesTest(equivalence.EquivalenceTest):

  def setUp(self):
    super().setUp()
    self._init_distr_cls(bates.Bates)
    self.total_count = 5.0
    self.low = -1.0
    self.high = 2.0
    self.assertion_fn = self.assertion_fn(rtol=1e-3)

  @parameterized.named_parameters(
      ('1d', (5.0, 0.0, 1.0)),
      # total_count must be static, so we only batch low/high
      ('2d', (5.0, np.zeros(2), np.ones(2))),
      ('rank 2', (5.0, np.zeros((3, 2)), np.ones((3, 2)))),
      ('broadcasted low', (5.0, 0.0, np.ones(3))),
      ('broadcasted high', (5.0, np.zeros(3), 1.0)),
  )
  def test_event_shape(self, distr_params):
    super()._test_event_shape(distr_params, dict())

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d, no shape', (5.0, 0.0, 1.0), ()),
      ('1d, int shape', (5.0, 0.0, 1.0), 1),
      ('1d, 1-tuple shape', (5.0, 0.0, 1.0), (1,)),
      ('1d, 2-tuple shape', (5.0, 0.0, 1.0), (2, 2)),
      ('2d, no shape', (5.0, np.zeros(2), np.ones(2)), ()),
      ('2d, 2-tuple shape', (5.0, np.zeros(2), np.ones(2)), (2, 2)),
      ('broadcasted', (5.0, np.zeros(3), 1.0), (2, 2)),
      # Batched total_count is not supported
  )
  def test_sample_shape(self, distr_params, sample_shape):
    super()._test_sample_shape(distr_params, dict(), sample_shape)

  @chex.all_variants
  @jax.numpy_rank_promotion('raise')
  @parameterized.named_parameters(
      ('1d, no shape', (5.0, 0.0, 1.0), ()),
      ('1d, 2-tuple shape', (5.0, 0.0, 1.0), (2, 2)),
      ('2d, no shape', (5.0, np.zeros(2), np.ones(2)), ()),
      ('2d, 2-tuple shape', (5.0, np.zeros(2), np.ones(2)), (2, 2)),
      # Batched total_count is not supported
  )
  def test_sample_and_log_prob(self, distr_params, sample_shape):
    super()._test_sample_and_log_prob(
        dist_args=distr_params,
        dist_kwargs=dict(),
        sample_shape=sample_shape,
        assertion_fn=self.assertion_fn,
    )

  @chex.all_variants
  @parameterized.named_parameters(
      ('log_prob', 'log_prob', (5.0, 0.0, 1.0)),
      ('prob', 'prob', (5.0, 0.0, 1.0)),
      ('cdf', 'cdf', (5.0, 0.0, 1.0)),
      ('log_cdf', 'log_cdf', (5.0, 0.0, 1.0)),
      ('survival_function', 'survival_function', (5.0, 0.0, 1.0)),
      ('log_survival_function', 'log_survival_function', (5.0, 0.0, 1.0)),
      ('n=1', 'prob', (1.0, 0.0, 1.0)),  # Should be uniform
      ('n=1 cdf', 'cdf', (1.0, -1.0, 2.0)),
      ('n=2', 'prob', (2.0, 0.0, 1.0)),
  )
  def test_method_with_inputs(self, function_string, distr_params):
    # Test inside and outside the support
    inputs = np.array([-1.0, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    super()._test_attribute(
        function_string,
        dist_args=distr_params,
        call_args=(inputs,),
        assertion_fn=self.assertion_fn,
    )

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      ('entropy', (5.0, 0.0, 1.0), 'entropy'),  # Entropy is not implemented
      ('mean', (5.0, 0.0, 1.0), 'mean'),
      ('variance', (5.0, 0.0, 1.0), 'variance'),
      ('stddev', (5.0, 0.0, 1.0), 'stddev'),
      ('mode', (5.0, 0.0, 1.0), 'mode'),
      # median is tested separately
      ('batched', (5.0, np.zeros(2), np.ones(2)), 'mean'),
      ('batched variance', (5.0, np.zeros(2), np.ones(2)), 'variance'),
  )
  def test_method(self, distr_params, function_string):
    # Entropy is not implemented for Bates in Distrax, so we expect
    # NotImplementedError.
    if function_string == 'entropy':
      with self.assertRaises(NotImplementedError):
        distrax_dist = self.distrax_cls(*distr_params)
        # Apply variant to the function call
        self.variant(getattr(distrax_dist, function_string))()  # pytype: disable=attribute-error
    else:
      super()._test_attribute(
          function_string, distr_params, assertion_fn=self.assertion_fn
      )

  @chex.all_variants(with_pmap=False)
  def test_median(self):
    dist = self.distrax_cls(5.0, 0.0, 1.0)
    self.assertion_fn(self.variant(dist.median)(), self.variant(dist.mean)())  # pytype: disable=attribute-error

    dist_batch = self.distrax_cls(5.0, np.zeros(2), np.ones(2))
    self.assertion_fn(self.variant(dist_batch.median)(), self.variant(dist_batch.mean)())  # pytype: disable=attribute-error

  @parameterized.named_parameters(
      ('total_count', 'total_count'),
      ('low', 'low'),
      ('high', 'high'),
  )
  def test_attribute(self, attribute_string):
    super()._test_attribute(
        attribute_string,
        dist_args=(self.total_count, self.low, self.high),
        assertion_fn=self.assertion_fn,
    )

  def test_jittable(self):
    super()._test_jittable(
        (self.total_count, self.low, self.high), assertion_fn=self.assertion_fn
    )

  @parameterized.named_parameters(
      ('single element', 2),
      ('range', slice(-1)),
      ('range_2', (slice(None), slice(-1))),
      ('ellipsis', (Ellipsis, -1)),
  )
  def test_slice(self, slice_):
    total_count = 5.0
    low = np.zeros((3, 4, 5))
    high = np.ones((3, 4, 5))
    dist = self.distrax_cls(total_count=total_count, low=low, high=high)

    # total_count is scalar, not sliced.
    self.assertion_fn(dist[slice_].total_count, total_count)
    self.assertion_fn(dist[slice_].low, low[slice_])
    self.assertion_fn(dist[slice_].high, high[slice_])

  def test_slice_different_parameterization(self):
    total_count = 5.0
    low = np.zeros((3, 4, 5))  # Batched
    high = 1.0  # Scalar
    dist = self.distrax_cls(total_count=total_count, low=low, high=high)

    self.assertion_fn(dist[..., -1].total_count, total_count)
    self.assertEqual(dist[..., -1].low.shape, (3, 4))
    self.assertion_fn(dist[..., -1].low, low[..., -1])  # Sliced
    self.assertEqual(dist[..., -1].high.shape, (3, 4))
    self.assertion_fn(dist[..., -1].high, high)  # Not slicing

  @chex.all_variants
  def test_n1_is_uniform(self):
    # Bates(n=1) is equivalent to Uniform
    # Do NOT JIT the constructor.
    unif = self.distrax_cls(total_count=1.0, low=self.low, high=self.high)

    # Get a variant-aware assertion function
    assertion_fn = self.assertion_fn

    # Check stats
    mean_val = self.variant(unif.mean)()  # JIT the method  # pytype: disable=attribute-error
    assertion_fn(mean_val, (self.low + self.high) / 2.0)

    variance_val = self.variant(unif.variance)()  # JIT the method  # pytype: disable=attribute-error
    assertion_fn(variance_val, (self.high - self.low) ** 2 / 12.0)

    # Check prob
    prob_fn = self.variant(unif.prob)  # JIT the method  # pytype: disable=attribute-error
    assertion_fn(prob_fn(self.low - 0.1), 0.0)
    assertion_fn(prob_fn(self.high + 0.1), 0.0)
    assertion_fn(prob_fn(mean_val), 1.0 / (self.high - self.low))

    # Check cdf
    cdf_fn = self.variant(unif.cdf)  # JIT the method  # pytype: disable=attribute-error
    assertion_fn(cdf_fn(self.low - 0.1), 0.0)
    assertion_fn(cdf_fn(self.high + 0.1), 1.0)
    assertion_fn(cdf_fn(mean_val), 0.5)
    assertion_fn(cdf_fn(self.low), 0.0)
    assertion_fn(cdf_fn(self.high), 1.0)


if __name__ == '__main__':
  absltest.main()
