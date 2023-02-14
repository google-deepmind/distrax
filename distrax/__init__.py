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
"""Distrax: Probability distributions in JAX."""

# Bijectors.
from distrax._src.bijectors.bijector import Bijector
from distrax._src.bijectors.bijector import BijectorLike
from distrax._src.bijectors.block import Block
from distrax._src.bijectors.chain import Chain
from distrax._src.bijectors.diag_linear import DiagLinear
from distrax._src.bijectors.diag_plus_low_rank_linear import DiagPlusLowRankLinear
from distrax._src.bijectors.gumbel_cdf import GumbelCDF
from distrax._src.bijectors.inverse import Inverse
from distrax._src.bijectors.lambda_bijector import Lambda
from distrax._src.bijectors.linear import Linear
from distrax._src.bijectors.lower_upper_triangular_affine import LowerUpperTriangularAffine
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.bijectors.rational_quadratic_spline import RationalQuadraticSpline
from distrax._src.bijectors.scalar_affine import ScalarAffine
from distrax._src.bijectors.shift import Shift
from distrax._src.bijectors.sigmoid import Sigmoid
from distrax._src.bijectors.split_coupling import SplitCoupling
from distrax._src.bijectors.tanh import Tanh
from distrax._src.bijectors.triangular_linear import TriangularLinear
from distrax._src.bijectors.unconstrained_affine import UnconstrainedAffine

# Distributions.
from distrax._src.distributions.bernoulli import Bernoulli
from distrax._src.distributions.beta import Beta
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.categorical_uniform import CategoricalUniform
from distrax._src.distributions.clipped import Clipped
from distrax._src.distributions.clipped import ClippedLogistic
from distrax._src.distributions.clipped import ClippedNormal
from distrax._src.distributions.deterministic import Deterministic
from distrax._src.distributions.dirichlet import Dirichlet
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.distribution import DistributionLike
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
from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from distrax._src.distributions.mvn_from_bijector import MultivariateNormalFromBijector
from distrax._src.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from distrax._src.distributions.mvn_tri import MultivariateNormalTri
from distrax._src.distributions.normal import Normal
from distrax._src.distributions.one_hot_categorical import OneHotCategorical
from distrax._src.distributions.quantized import Quantized
from distrax._src.distributions.softmax import Softmax
from distrax._src.distributions.straight_through import straight_through_wrapper
from distrax._src.distributions.transformed import Transformed
from distrax._src.distributions.uniform import Uniform
from distrax._src.distributions.von_mises import VonMises

# Utilities.
from distrax._src.utils.conversion import as_bijector
from distrax._src.utils.conversion import as_distribution
from distrax._src.utils.conversion import to_tfp
from distrax._src.utils.hmm import HMM
from distrax._src.utils.importance_sampling import importance_sampling_ratios
from distrax._src.utils.math import multiply_no_nan
from distrax._src.utils.monte_carlo import estimate_kl_best_effort
from distrax._src.utils.monte_carlo import mc_estimate_kl
from distrax._src.utils.monte_carlo import mc_estimate_kl_with_reparameterized
from distrax._src.utils.monte_carlo import mc_estimate_mode
from distrax._src.utils.transformations import register_inverse

__version__ = "0.1.3"

__all__ = (
    "as_bijector",
    "as_distribution",
    "Bernoulli",
    "Beta",
    "Bijector",
    "BijectorLike",
    "Block",
    "Categorical",
    "CategoricalUniform",
    "Chain",
    "Clipped",
    "ClippedLogistic",
    "ClippedNormal",
    "Deterministic",
    "DiagLinear",
    "DiagPlusLowRankLinear",
    "Dirichlet",
    "Distribution",
    "DistributionLike",
    "EpsilonGreedy",
    "estimate_kl_best_effort",
    "Gamma",
    "Greedy",
    "Gumbel",
    "GumbelCDF",
    "HMM",
    "importance_sampling_ratios",
    "Independent",
    "Inverse",
    "Joint",
    "Lambda",
    "Laplace",
    "Linear",
    "Logistic",
    "LogStddevNormal",
    "LowerUpperTriangularAffine",
    "MaskedCoupling",
    "mc_estimate_kl",
    "mc_estimate_kl_with_reparameterized",
    "mc_estimate_mode",
    "MixtureOfTwo",
    "MixtureSameFamily",
    "Multinomial",
    "multiply_no_nan",
    "MultivariateNormalDiag",
    "MultivariateNormalDiagPlusLowRank",
    "MultivariateNormalFromBijector",
    "MultivariateNormalFullCovariance",
    "MultivariateNormalTri",
    "Normal",
    "OneHotCategorical",
    "Quantized",
    "RationalQuadraticSpline",
    "register_inverse",
    "ScalarAffine",
    "Shift",
    "Sigmoid",
    "Softmax",
    "SplitCoupling",
    "straight_through_wrapper",
    "Tanh",
    "to_tfp",
    "Transformed",
    "TriangularLinear",
    "UnconstrainedAffine",
    "Uniform",
    "VonMises",
)


#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Distrax public API. /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
