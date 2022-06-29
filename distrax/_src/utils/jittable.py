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
import jax.numpy as jnp
import numpy as np


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
    jax_args, py_args = _split_jax_and_python_list(self._args)
    jax_kwargs, py_kwargs = _split_jax_and_python_dict(self._kwargs)
    jax_state, py_state = _split_jax_and_python_dict(self.__dict__)
    return ((jax_args, jax_kwargs, jax_state), (py_args, py_kwargs, py_state))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    py_args, py_kwargs, py_state = aux_data
    jax_args, jax_kwargs, jax_state = children
    args = _join_jax_and_python_list(jax_args, *py_args)
    kwargs = _join_jax_and_python_dict(jax_kwargs, *py_kwargs)
    state = _join_jax_and_python_dict(jax_state, *py_state)
    obj = cls(*args, **kwargs)
    obj.__dict__ = state
    return obj


def _is_jax_data(x):
  return all(
      isinstance(a, (jnp.ndarray, np.ndarray)) for a in jax.tree_leaves(x))


def _split_jax_and_python_list(obj):
  is_jax = list(map(_is_jax_data, obj))
  jax_data = [v if j else None for v, j in zip(obj, is_jax)]
  py_data = [v if not j else None for v, j in zip(obj, is_jax)]
  return jax_data, (py_data, is_jax)


def _split_jax_and_python_dict(obj):
  is_jax = {k: _is_jax_data(v) for k, v in obj.items()}
  jax_data = {k: v if is_jax[k] else None for k, v in obj.items()}
  py_data = {k: v if not is_jax[k] else None for k, v in obj.items()}
  return jax_data, (py_data, is_jax)


def _join_jax_and_python_list(jax_data, py_data, is_jax):
  return [j if is_jax else p for j, p, is_jax in zip(jax_data, py_data, is_jax)]


def _join_jax_and_python_dict(jax_data, py_data, is_jax):
  return {k: jax_data[k] if v else py_data[k] for k, v in is_jax.items()}
