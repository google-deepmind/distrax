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
"""Distrax distributions."""

from distrax._src.distributions.bernoulli import Bernoulli
from distrax._src.distributions.beta import Beta
from distrax._src.distributions.binomial import Binomial
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.categorical_uniform import CategoricalUniform
from distrax._src.distributions.clipped import Clipped
from distrax._src.distributions.deterministic import Deterministic
from distrax._src.distributions.dirichlet import Dirichlet
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.distribution_from_tfp import DistributionFromTfp
from distrax._src.distributions.epsilon_greedy import EpsilonGreedy
from distrax._src.distributions.gamma import Gamma
from distrax._src.distributions.greedy import Greedy
from distrax._src.distributions.gumbel import Gumbel
from distrax._src.distributions.independent import Independent
from distrax._src.distributions.joint import Joint
from distrax._src.distributions.laplace import Laplace
from distrax._src.distributions.log_stddev_normal import LogStddevNormal
from distrax._src.distributions.logistic import Logistic
from distrax._src.distributions.mixture_of_two import MixtureOfTwo
from distrax._src.distributions.mixture_same_family import MixtureSameFamily
from distrax._src.distributions.multinomial import Multinomial
from distrax._src.distributions.mvn_diag import MvnDiag
from distrax._src.distributions.mvn_diag_plus_low_rank import MvnDiagPlusLowRank
from distrax._src.distributions.mvn_from_bijector import MvnFromBijector
from distrax._src.distributions.mvn_full_covariance import MvnFullCovariance
from distrax._src.distributions.mvn_tri import MvnTri
from distrax._src.distributions.normal import Normal
from distrax._src.distributions.one_hot_categorical import OneHotCategorical
from distrax._src.distributions.quantized import Quantized
from distrax._src.distributions.softmax import Softmax
from distrax._src.distributions.straight_through import StraightThrough
from distrax._src.distributions.tfp_compatible_distribution import TfpCompatibleDistribution
from distrax._src.distributions.transformed import Transformed
from distrax._src.distributions.uniform import Uniform
from distrax._src.distributions.von_mises import VonMises

__all__ = (
    'Bernoulli',
    'Beta',
    'Binomial',
    'Categorical',
    'CategoricalUniform',
    'Clipped',
    'Deterministic',
    'Dirichlet',
    'Distribution',
    'DistributionFromTfp',
    'EpsilonGreedy',
    'Gamma',
    'Greedy',
    'Gumbel',
    'Independent',
    'Joint',
    'Laplace',
    'LogStddevNormal',
    'Logistic',
    'MixtureOfTwo',
    'MixtureSameFamily',
    'Multinomial',
    'MvnDiag',
    'MvnDiagPlusLowRank',
    'MvnFromBijector',
    'MvnFullCovariance',
    'MvnTri',
    'Normal',
    'OneHotCategorical',
    'Quantized',
    'Softmax',
    'StraightThrough',
    'TfpCompatibleDistribution',
    'Transformed',
    'Uniform',
    'VonMises',
)
