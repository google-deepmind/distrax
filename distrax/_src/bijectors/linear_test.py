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
"""Tests for `linear.py`."""

from absl.testing import absltest
from absl.testing import parameterized
from distrax._src.bijectors import linear
import jax.numpy as jnp


class MockLinear(linear.Linear):

  def forward_and_log_det(self, x):
    raise Exception  # pylint:disable=broad-exception-raised


class LinearTest(parameterized.TestCase):

  @parameterized.parameters(
      {'event_dims': 1, 'batch_shape': (), 'dtype': jnp.float16},
      {'event_dims': 10, 'batch_shape': (2, 3), 'dtype': jnp.float32})
  def test_properties(self, event_dims, batch_shape, dtype):
    bij = MockLinear(event_dims, batch_shape, dtype)
    self.assertEqual(bij.event_ndims_in, 1)
    self.assertEqual(bij.event_ndims_out, 1)
    self.assertTrue(bij.is_constant_jacobian)
    self.assertTrue(bij.is_constant_log_det)
    self.assertEqual(bij.event_dims, event_dims)
    self.assertEqual(bij.batch_shape, batch_shape)
    self.assertEqual(bij.dtype, dtype)
    with self.assertRaises(NotImplementedError):
      bij.matrix  # pylint: disable=pointless-statement


if __name__ == '__main__':
  absltest.main()
