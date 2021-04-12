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
"""Tests for `bijector.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from distrax._src.bijectors import bijector
import jax
import jax.numpy as jnp
import numpy as np


class DummyBijector(bijector.Bijector):

  def forward_and_log_det(self, x):
    super()._check_forward_input_shape(x)
    return x, jnp.zeros(x.shape[:-1], jnp.float_)

  def inverse_and_log_det(self, y):
    super()._check_inverse_input_shape(y)
    return y, jnp.zeros(y.shape[:-1], jnp.float_)


class BijectorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('negative ndims_in', -1, 1, False, False),
      ('negative ndims_out', 1, -1, False, False),
      ('non-consistent constant properties', 1, 1, True, False),
  )
  def test_invalid_parameters(self, ndims_in, ndims_out, cnst_jac, cnst_logdet):
    with self.assertRaises(ValueError):
      DummyBijector(ndims_in, ndims_out, cnst_jac, cnst_logdet)

  @chex.all_variants
  @parameterized.parameters('forward', 'inverse')
  def test_invalid_inputs(self, method_str):
    bij = DummyBijector(1, 1, True, True)
    fn = self.variant(getattr(bij, method_str))
    with self.assertRaises(ValueError):
      fn(jnp.zeros(()))

  def test_jittable(self):
    @jax.jit
    def forward(bij, x):
      return bij.forward(x)

    bij = DummyBijector(1, 1, True, True)
    x = jnp.zeros((4,))
    np.testing.assert_allclose(forward(bij, x), x)


if __name__ == '__main__':
  absltest.main()
