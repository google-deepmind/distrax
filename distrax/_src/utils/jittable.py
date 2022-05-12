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
from typing import Tuple

import jax


class Jittable(metaclass=abc.ABCMeta):
  """ABC that can be passed as an arg to a jitted fn, with readable state.

  Subclasses that intend for array-like fields to be treated as JAX data should
  expose the names of those fields by overriding `_pytree_fields`. All other
  fields will be treated as non-jittable metadata.
  """

  def __new__(cls, *args, **kwargs):
    try:
      registered_cls = jax.tree_util.register_pytree_node_class(cls)
    except ValueError:
      registered_cls = cls  # already registered
    instance = super(Jittable, cls).__new__(registered_cls)
    instance._args = args
    instance._kwargs = kwargs
    return instance

  def _pytree_fields(self) -> Tuple[str, ...]:
    """Tuple of strings indicating the object's properties that are JAX data.

    All other properties of the object are considered non-jittable metadata, and
    will not be recursively traversed during JAX serialization. In addition, any
    metadata properties not flagged by this method will be considered static
    with respect to compiled JAX functions: failing to flag a jittable data
    property in this way may result in unnecessary recompilations, exceptions,
    or leaked tracers.

    Returns:
      Tuple of strings indicating the fields to consider as JAX data.
    """
    return ()

  def tree_flatten(self):
    pytree_data = {k: v for k, v in self.__dict__.items()
                   if k in self._pytree_fields()}
    metadata = {k: v for k, v in self.__dict__.items()
                if k not in self._pytree_fields()}
    return ((pytree_data,), ((self._args, self._kwargs), metadata))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    (args, kwargs), metadata = aux_data
    (pytree_data,) = children
    obj = cls(*args, **kwargs)
    obj.__dict__.update(metadata)
    obj.__dict__.update(pytree_data)
    return obj
