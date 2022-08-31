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
"""Tests for `joint.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.joint import Joint
from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.normal import Normal
import jax
import jax.numpy as jnp
import numpy as np
import tree


def _make_nested_distributions_and_inputs(batch_shape=None, shift=0.0):
  distributions = dict(
      categoricals=[
          Categorical(np.array([0 + shift, 1, 2])),
          Categorical(np.array([2 + shift, 0, 1])),
      ],
      normals=(
          Normal(np.array(1) + shift, np.array(2) + shift),
          Normal(np.array(2) + shift, np.array(1) + shift),
          Normal(np.array(5) + shift, np.array(4) + shift),
      ),
      multivariate=MultivariateNormalDiag(np.array([5, 6, 7]) + shift,
                                          np.array([3, 3, 3]) + shift))

  inputs = dict(
      categoricals=[
          np.array(0),
          np.array(1),
      ],
      normals=(
          np.array(2.5),
          np.array(3.1),
          np.array(4.2),
      ),
      multivariate=np.array([5.1, 5.2, 5.3]))

  if isinstance(batch_shape, tuple):
    def _add_batch_shape_to_tensor(x):
      for dim in reversed(batch_shape):
        x = jnp.repeat(x[None, ...], dim, axis=0)
      return x

    def _add_batch_shape_to_distribution(d):
      if isinstance(d, Categorical):
        return Categorical(_add_batch_shape_to_tensor(d.logits_parameter()))
      elif isinstance(d, Normal):
        return Normal(_add_batch_shape_to_tensor(d.loc),
                      _add_batch_shape_to_tensor(d.scale))
      elif isinstance(d, MultivariateNormalDiag):
        return MultivariateNormalDiag(_add_batch_shape_to_tensor(d.loc),
                                      _add_batch_shape_to_tensor(d.scale_diag))

    distributions = tree.map_structure(_add_batch_shape_to_distribution,
                                       distributions)
    inputs = tree.map_structure(_add_batch_shape_to_tensor, inputs)

  return distributions, inputs


class JointTest(parameterized.TestCase):

  @chex.all_variants
  @parameterized.named_parameters(
      ('categorical', Categorical, (np.array([0, 1, 2]),), np.array(0)),
      ('normal', Normal, (np.array([0, 1, 2]), np.array([1, 2, 3])),
       np.array([0, 0, 0])),
      ('mvn', MultivariateNormalDiag,
       (np.array([[0, 1, 2], [3, 4, 5]]), np.array([[1, 2, 3], [4, 5, 6]])),
       np.array([[0, 0, 0], [0, 0, 0]])),
  )
  def test_single_distribution(self, fn, params, x):
    dist = fn(*params)
    joint = Joint(dist)

    key = jax.random.PRNGKey(0)
    subkey, = jax.random.split(key, 1)

    with self.subTest('sample'):
      actual = self.variant(joint.sample)(seed=key)
      expected = dist.sample(seed=subkey)
      np.testing.assert_allclose(actual, expected, rtol=1e-6)

    with self.subTest('log_prob'):
      actual = self.variant(joint.log_prob)(x)
      expected = dist.log_prob(x)
      np.testing.assert_allclose(actual, expected, rtol=3e-5)

    with self.subTest('sample_and_log_prob'):
      actual_sample, actual_log_prob = self.variant(joint.sample_and_log_prob)(
          seed=key)
      expected_sample, expected_log_prob = dist.sample_and_log_prob(seed=subkey)
      np.testing.assert_allclose(actual_sample, expected_sample, rtol=3e-5)
      np.testing.assert_allclose(actual_log_prob, expected_log_prob, rtol=3e-5)

  @chex.all_variants
  def test_distribution_tuple(self):
    distributions = (
        Categorical(np.array([0, 1, 2])),
        MultivariateNormalDiag(np.array([1, 2, 3]), np.array([2, 3, 4])))
    inputs = (np.array(0), np.array([0.1, 0.2, 0.3]))
    joint = Joint(distributions)

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 2)

    with self.subTest('sample'):
      actuals = self.variant(joint.sample)(seed=key)
      assert isinstance(actuals, tuple)
      for actual, dist, subkey in zip(actuals, distributions, subkeys):
        expected = dist.sample(seed=subkey)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    with self.subTest('log_prob'):
      actual = self.variant(joint.log_prob)(inputs)
      log_probs = [dist.log_prob(x) for dist, x in zip(distributions, inputs)]
      expected = sum(log_probs)
      np.testing.assert_array_equal(actual, expected)

    with self.subTest('sample_and_log_prob'):
      actual_sample, actual_log_prob = self.variant(joint.sample_and_log_prob)(
          seed=key)
      assert isinstance(actual_sample, tuple)
      samples = []
      log_probs = []
      for dist, subkey in zip(distributions, subkeys):
        sample, log_prob = dist.sample_and_log_prob(seed=subkey)
        samples.append(sample)
        log_probs.append(log_prob)
      expected_sample = tuple(samples)
      expected_log_prob = sum(log_probs)
      for actual, expected in zip(actual_sample, expected_sample):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
      np.testing.assert_array_equal(actual_log_prob, expected_log_prob)

  @chex.all_variants
  def test_distribution_list(self):
    distributions = [
        Categorical(np.array([0, 1, 2])),
        MultivariateNormalDiag(np.array([1, 2, 3]), np.array([2, 3, 4]))]
    inputs = [np.array(0), np.array([0.1, 0.2, 0.3])]
    joint = Joint(distributions)

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 2)

    with self.subTest('sample'):
      actuals = self.variant(joint.sample)(seed=key)
      assert isinstance(actuals, list)
      for actual, dist, subkey in zip(actuals, distributions, subkeys):
        expected = dist.sample(seed=subkey)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    with self.subTest('log_prob'):
      actual = self.variant(joint.log_prob)(inputs)
      log_probs = [dist.log_prob(x) for dist, x in zip(distributions, inputs)]
      expected = sum(log_probs)
      np.testing.assert_array_equal(actual, expected)

    with self.subTest('sample_and_log_prob'):
      actual_sample, actual_log_prob = self.variant(joint.sample_and_log_prob)(
          seed=key)
      assert isinstance(actual_sample, list)
      expected_sample = []
      log_probs = []
      for dist, subkey in zip(distributions, subkeys):
        sample, log_prob = dist.sample_and_log_prob(seed=subkey)
        expected_sample.append(sample)
        log_probs.append(log_prob)
      expected_log_prob = sum(log_probs)
      for actual, expected in zip(actual_sample, expected_sample):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
      np.testing.assert_array_equal(actual_log_prob, expected_log_prob)

  @chex.all_variants
  def test_distributions_with_batch_shape(self):
    distributions = [
        Categorical(np.array([[0, 1, 2], [3, 4, 5]])),
        MultivariateNormalDiag(
            np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]),
            np.array([[1, 2, 3, 5, 6], [2, 3, 4, 5, 6]]))]
    inputs = [np.array([0, 1]), np.zeros((2, 5))]
    joint = Joint(distributions)
    assert joint.batch_shape == distributions[0].batch_shape
    assert joint.batch_shape == distributions[1].batch_shape

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 2)

    with self.subTest('sample'):
      actuals = self.variant(joint.sample)(seed=key)
      assert isinstance(actuals, list)
      assert actuals[0].shape == (2,)
      assert actuals[1].shape == (2, 5)
      for actual, dist, subkey in zip(actuals, distributions, subkeys):
        expected = dist.sample(seed=subkey)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    with self.subTest('log_prob'):
      actual = self.variant(joint.log_prob)(inputs)
      assert actual.shape == (2,)
      log_probs = [dist.log_prob(x) for dist, x in zip(distributions, inputs)]
      expected = sum(log_probs)
      np.testing.assert_allclose(actual, expected, rtol=1e-6)

    with self.subTest('sample_and_log_prob'):
      actual_sample, actual_log_prob = self.variant(joint.sample_and_log_prob)(
          seed=key)
      assert isinstance(actual_sample, list)
      assert actual_sample[0].shape == (2,)
      assert actual_sample[1].shape == (2, 5)
      assert actual_log_prob.shape == (2,)
      expected_sample = []
      log_probs = []
      for dist, subkey in zip(distributions, subkeys):
        sample, log_prob = dist.sample_and_log_prob(seed=subkey)
        expected_sample.append(sample)
        log_probs.append(log_prob)
      expected_log_prob = sum(log_probs)
      for actual, expected in zip(actual_sample, expected_sample):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
      np.testing.assert_allclose(actual_log_prob, expected_log_prob, rtol=1e-6)

  @chex.all_variants
  def test_nested_distributions(self):
    distributions, inputs = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 6)

    with self.subTest('sample'):
      actuals = self.variant(joint.sample)(seed=key)
      assert isinstance(actuals, dict)
      assert isinstance(actuals['categoricals'], list)
      assert isinstance(actuals['normals'], tuple)
      assert isinstance(actuals['multivariate'], jnp.ndarray)

      flat_actuals = tree.flatten(actuals)
      flat_dists = tree.flatten(distributions)
      for actual, dist, subkey in zip(flat_actuals, flat_dists, subkeys):
        expected = dist.sample(seed=subkey)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    with self.subTest('log_prob'):
      actual = self.variant(joint.log_prob)(inputs)
      flat_dists = tree.flatten(distributions)
      flat_inputs = tree.flatten(inputs)
      log_probs = [dist.log_prob(x) for dist, x in zip(flat_dists, flat_inputs)]
      expected = sum(log_probs)
      np.testing.assert_array_equal(actual, expected)

    with self.subTest('sample_and_log_prob'):
      actual_sample, actual_log_prob = self.variant(joint.sample_and_log_prob)(
          seed=key)
      assert isinstance(actual_sample, dict)
      assert isinstance(actual_sample['categoricals'], list)
      assert isinstance(actual_sample['normals'], tuple)
      assert isinstance(actual_sample['multivariate'], jnp.ndarray)

      expected_sample = []
      log_probs = []
      flat_dists = tree.flatten(distributions)
      for dist, subkey in zip(flat_dists, subkeys):
        sample, log_prob = dist.sample_and_log_prob(seed=subkey)
        expected_sample.append(sample)
        log_probs.append(log_prob)
      expected_log_prob = sum(log_probs)
      flat_actuals = tree.flatten(actual_sample)
      for actual, expected in zip(flat_actuals, expected_sample):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
      np.testing.assert_allclose(actual_log_prob, expected_log_prob, rtol=1e-6)

  @chex.all_variants(with_pmap=False)
  def test_entropy(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)
    actual = self.variant(joint.entropy)()
    flat_dists = tree.flatten(distributions)
    expected = sum(dist.entropy() for dist in flat_dists)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

  @chex.all_variants(with_pmap=False)
  def test_mode(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)
    actual = self.variant(joint.mode)()
    expected = tree.map_structure(lambda d: d.mode(), distributions)
    for actual, expected in zip(tree.flatten(actual), tree.flatten(expected)):
      np.testing.assert_array_equal(actual, expected)

  @chex.all_variants(with_pmap=False)
  def test_mean(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    del distributions['categoricals']  # Mean is not defined for these.
    joint = Joint(distributions)
    actual = self.variant(joint.mean)()
    expected = tree.map_structure(lambda d: d.mean(), distributions)
    for actual, expected in zip(tree.flatten(actual), tree.flatten(expected)):
      np.testing.assert_array_equal(actual, expected)

  @chex.all_variants(with_pmap=False)
  def test_median(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    del distributions['categoricals']  # Median is not defined for these.
    joint = Joint(distributions)
    actual = self.variant(joint.median)()
    expected = tree.map_structure(lambda d: d.median(), distributions)
    for actual, expected in zip(tree.flatten(actual), tree.flatten(expected)):
      np.testing.assert_array_equal(actual, expected)

  @chex.all_variants
  def test_kl_divergence(self):
    dists_a, _ = _make_nested_distributions_and_inputs(shift=0.0)
    dists_b, _ = _make_nested_distributions_and_inputs(shift=1.0)

    joint_a = Joint(dists_a)
    joint_b = Joint(dists_b)
    actual = self.variant(joint_a.kl_divergence)(joint_b)

    kls = []
    for dist_a, dist_b in zip(tree.flatten(dists_a), tree.flatten(dists_b)):
      kls.append(dist_a.kl_divergence(dist_b))
    expected = sum(kls)

    np.testing.assert_allclose(actual, expected, rtol=1e-4)

  @chex.all_variants
  def test_log_cdf(self):
    distributions, inputs = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)
    actual = self.variant(joint.log_cdf)(inputs)
    flat_dists = tree.flatten(distributions)
    flat_inputs = tree.flatten(inputs)
    expected = sum(dist.log_cdf(x) for dist, x in zip(flat_dists, flat_inputs))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

  def test_distributions_property(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)
    tree.assert_same_structure(joint.distributions, distributions)

  def test_event_shape_property(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)
    all_event_shapes = joint.event_shape
    for dist, event_shape in zip(tree.flatten(distributions),
                                 tree.flatten_up_to(distributions,
                                                    all_event_shapes)):
      np.testing.assert_equal(dist.event_shape, event_shape)

  def test_dtype_property(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    joint = Joint(distributions)
    all_dtypes = joint.dtype
    for dist, dtype in zip(tree.flatten(distributions),
                           tree.flatten(all_dtypes)):
      np.testing.assert_equal(dist.dtype, dtype)

  @chex.all_variants
  @parameterized.named_parameters(
      ('1d batch first element', (10,), 0),
      ('1d batch last element', (10,), -1),
      ('1d batch first two elements', (10,), slice(0, 2)),
      ('1d batch first and third elements', (10,), slice(0, 4, 2)),
      ('2d batch first element', (10, 7), 0),
      ('2d batch last element', (10, 7), -1),
      ('2d batch first two elements', (10, 7), slice(0, 2)),
      ('2d batch first and third elements', (10, 7), slice(0, 4, 2)),
  )
  def test_indexing(self, batch_shape, index):
    distributions, inputs = _make_nested_distributions_and_inputs(
        batch_shape=batch_shape)
    inputs = tree.map_structure(lambda x: x[index], inputs)

    joint = Joint(distributions)
    joint_indexed = joint[index]

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 6)

    with self.subTest('batch shape'):
      for dist, indexed in zip(tree.flatten(distributions),
                               tree.flatten(joint_indexed.distributions)):
        assert dist[index].batch_shape == indexed.batch_shape

    with self.subTest('event shape'):
      for dist, indexed in zip(tree.flatten(distributions),
                               tree.flatten(joint_indexed.distributions)):
        assert dist[index].event_shape == indexed.event_shape

    with self.subTest('sample'):
      all_samples = self.variant(joint_indexed.sample)(seed=key)
      for dist, subkey, actual in zip(tree.flatten(distributions),
                                      subkeys,
                                      tree.flatten(all_samples)):
        expected = dist[index].sample(seed=subkey)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    with self.subTest('sample_and_log_prob'):
      actual_samples, actual_log_probs = self.variant(
          joint_indexed.sample_and_log_prob)(seed=key)
      expected_outputs = [
          dist[index].sample_and_log_prob(seed=subkey)
          for dist, subkey in zip(tree.flatten(distributions), subkeys)]
      expected_samples = [sample for sample, _ in expected_outputs]
      expected_log_probs = sum(lp for _, lp in expected_outputs)
      for actual, expected in zip(tree.flatten(actual_samples),
                                  expected_samples):
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
      np.testing.assert_allclose(
          actual_log_probs, expected_log_probs, rtol=1e-6)

    with self.subTest('log_prob'):
      actual = self.variant(joint_indexed.log_prob)(inputs)
      expected = sum(dist[index].log_prob(x) for dist, x in zip(
          tree.flatten(distributions), tree.flatten(inputs)))
      np.testing.assert_allclose(actual, expected, rtol=1e-6)

  def test_raise_on_mismatched_batch_shape(self):
    distributions = dict(
        unbatched=Categorical(np.zeros((3,))),
        batched=Normal(np.zeros((3, 4, 5)), np.ones((3, 4, 5))))

    with self.assertRaises(ValueError):
      Joint(distributions)

  @chex.all_variants
  def test_raise_on_incompatible_distributions_kl(self):
    distributions, _ = _make_nested_distributions_and_inputs()
    incompatible = dict(
        categoricals=distributions['normals'],
        normals=distributions['categoricals'],
        multivariate=distributions['multivariate'])

    joint_a = Joint(distributions)
    joint_b = Joint(incompatible)

    with self.assertRaises(ValueError):
      self.variant(joint_a.kl_divergence)(joint_b)


if __name__ == '__main__':
  absltest.main()
