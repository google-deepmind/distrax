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
"""Abstract class for Jittable objects."""

import abc
import jax


class Jittable(metaclass=abc.ABCMeta):
  """ABC that can be passed as an arg to a jitted fn, with readable state."""

  def __new__(cls, *args, **kwargs):
    try:
      registered_cls = jax.tree_util.register_pytree_node_class(cls)
    except ValueError:
      registered_cls = cls  # already registered
    instance = super(Jittable, cls).__new__(registered_cls)
    instance._args = args
    instance._kwargs = kwargs
    return instance

  def tree_flatten(self):
    jax_data, python_data = _partition_by_jittability(self.__dict__)
    return ((jax_data,), ((self._args, self._kwargs), python_data))

  @classmethod
  def tree_unflatten(cls, aux_data, jax_data):
    (args, kwargs), python_data = aux_data
    obj = cls(*args, **kwargs)
    obj.__dict__ = python_data
    obj.__dict__.update(jax_data[0])
    return obj


def _is_jax_data(x):
  if isinstance(x, jax.numpy.ndarray):
    return True
  elif hasattr(x, 'tree_flatten') and hasattr(x, 'tree_unflatten'):
    return True
  else:
    return False


def _partition_by_jittability(data):
  """Partitions the dictionary into jittable data and pure-python data."""
  jax_data = {}
  python_data = {}

  for k, v in data.items():
    if all(map(_is_jax_data, jax.tree_leaves(v))):
      jax_data[k] = v
    else:
      python_data[k] = v

  return jax_data, python_data
