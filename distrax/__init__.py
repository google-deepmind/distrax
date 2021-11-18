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
from distrax._src.bijectors.inverse import Inverse
from distrax._src.bijectors.lambda_bijector import Lambda
from distrax._src.bijectors.lower_upper_triangular_affine import LowerUpperTriangularAffine
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.bijectors.rational_quadratic_spline import RationalQuadraticSpline
from distrax._src.bijectors.scalar_affine import ScalarAffine
from distrax._src.bijectors.sigmoid import Sigmoid
from distrax._src.bijectors.split_coupling import SplitCoupling
from distrax._src.bijectors.tanh import Tanh
from distrax._src.bijectors.unconstrained_affine import UnconstrainedAffine

# Distributions.
from distrax._src.distributions.bernoulli import Bernoulli
from distrax._src.distributions.categorical import Categorical
from distrax._src.distributions.deterministic import Deterministic
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.distribution import DistributionLike
from distrax._src.distributions.epsilon_greedy import EpsilonGreedy
from distrax._src.distributions.gamma import Gamma
from distrax._src.distributions.greedy import Greedy
from distrax._src.distributions.independent import Independent
from distrax._src.distributions.laplace import Laplace
from distrax._src.distributions.log_stddev_normal import LogStddevNormal
from distrax._src.distributions.logistic import Logistic
from distrax._src.distributions.mixture_same_family import MixtureSameFamily
from distrax._src.distributions.multinomial import Multinomial
from distrax._src.distributions.mvn_diag import MultivariateNormalDiag
from distrax._src.distributions.normal import Normal
from distrax._src.distributions.one_hot_categorical import OneHotCategorical
from distrax._src.distributions.quantized import Quantized
from distrax._src.distributions.softmax import Softmax
from distrax._src.distributions.transformed import Transformed
from distrax._src.distributions.uniform import Uniform

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

__version__ = "0.0.3"

__all__ = (
    "as_bijector",
    "as_distribution",
    "Bernoulli",
    "Bijector",
    "BijectorLike",
    "Block",
    "Categorical",
    "Chain",
    "Distribution",
    "DistributionLike",
    "EpsilonGreedy",
    "estimate_kl_best_effort",
    "Gamma",
    "Greedy",
    "HMM",
    "importance_sampling_ratios",
    "Independent",
    "Inverse",
    "Lambda",
    "Laplace",
    "LogStddevNormal",
    "Logistic",
    "LowerUpperTriangularAffine",
    "MaskedCoupling",
    "mc_estimate_kl",
    "mc_estimate_kl_with_reparameterized",
    "mc_estimate_mode",
    "MixtureSameFamily",
    "Multinomial",
    "multiply_no_nan",
    "MultivariateNormalDiag",
    "Normal",
    "OneHotCategorical",
    "Quantized",
    "RationalQuadraticSpline",
    "register_inverse",
    "ScalarAffine",
    "Sigmoid",
    "Softmax",
    "SplitCoupling",
    "to_tfp",
    "Transformed",
    "UnconstrainedAffine",
    "Uniform",
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
