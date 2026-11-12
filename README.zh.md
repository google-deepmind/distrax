# Distrax (基于 JAX 的概率分布与双射变换库)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

![CI status](https://github.com/deepmind/distrax/workflows/tests/badge.svg)
![pypi](https://img.shields.io/pypi/v/distrax)

Distrax 是一个轻量级的概率分布（probability distributions）与双射变换（bijectors）库。它是 [TensorFlow Probability](https://www.tensorflow.org/probability) (TFP) 子集的 JAX 原生重新实现，并引入了部分新特性且着重强调可扩展性。

## 安装指南 (Installation)

您可以通过 PyPI 安装 Distrax 的最新正式发布版本：

```sh
pip install distrax
```

或者从 GitHub 安装最新的开发版本：

```sh
pip install git+https://github.com/deepmind/distrax.git
```

如需运行测试或[示例程序](https://github.com/deepmind/distrax/tree/master/examples)，您需要安装额外的[依赖项](https://github.com/deepmind/distrax/tree/master/requirements)。

## 设计原则 (Design Principles)

DeepMind JAX 生态系统的通用设计原则已在[此博文](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research)中阐述。此外，Distrax 特别强调以下几点：

1. **可读性（Readability）**：Distrax 的实现旨在自包含，并尽可能贴近底层的数学公式表达。
2. **可扩展性（Extensibility）**：我们尽可能简化了用户定义自定义概率分布或双射器的流程。这在强化学习（RL）等领域尤为实用，例如用户可能希望为概率化智能体策略定义自定义行为。
3. **兼容性（Compatibility）**：Distrax 并非旨在替代 TFP，TFP 包含了许多我们无意复制的高级特性。为此，我们使分布与双射器的 API 尽可能实现跨框架兼容，并提供了在等效的 Distrax 和 TFP 类之间进行相互转换的实用工具。

## 功能特性 (Features)

### 概率分布 (Distributions)

在 Distrax 中定义和使用概率分布非常简单，尤其是对熟悉 TFP 的用户而言。让我们并排对比两者：

```python
import distrax
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

key = jax.random.PRNGKey(1234)
mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.1, 0.2, 0.3])

dist_distrax = distrax.MultivariateNormalDiag(mu, sigma)
dist_tfp = tfd.MultivariateNormalDiag(mu, sigma)

samples = dist_distrax.sample(seed=key)

# 两者均输出 1.775
print(dist_distrax.log_prob(samples))
print(dist_tfp.log_prob(samples))
```

除了行为保持一致外，Distrax 分布与 TFP 分布还具有跨框架兼容性。例如：

```python
mu_0 = jnp.array([-1., 0., 1.])
sigma_0 = jnp.array([0.1, 0.2, 0.3])
dist_distrax = distrax.MultivariateNormalDiag(mu_0, sigma_0)

mu_1 = jnp.array([1., 2., 3.])
sigma_1 = jnp.array([0.2, 0.3, 0.4])
dist_tfp = tfd.MultivariateNormalDiag(mu_1, sigma_1)

# 两者均输出 85.237
print(dist_distrax.kl_divergence(dist_tfp))
print(tfd.kl_divergence(dist_distrax, dist_tfp))
```

Distrax 分布实现了 `sample_and_log_prob` 方法，可在一行代码中同时获取采样样本及其对数概率。对于某些分布而言，这比分别调用 `sample` 和 `log_prob` 更加高效：

```python
mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.1, 0.2, 0.3])
dist_distrax = distrax.MultivariateNormalDiag(mu, sigma)

samples = dist_distrax.sample(seed=key, sample_shape=())
log_prob = dist_distrax.log_prob(samples)

# 上述操作的单行等效形式：
samples, log_prob = dist_distrax.sample_and_log_prob(seed=key, sample_shape=())
```

TFP 分布可以直接作为输入传递给 Distrax 元分布（meta-distributions）。例如：

```python
key = jax.random.PRNGKey(1234)

mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.2, 0.3, 0.4])
dist_tfp = tfd.Normal(mu, sigma)

metadist_distrax = distrax.Independent(dist_tfp, reinterpreted_batch_ndims=1)
samples = metadist_distrax.sample(seed=key)
print(metadist_distrax.log_prob(samples))  # 输出 0.38871175
```

若要在 TFP 元分布中使用 Distrax 分布，Distrax 提供了包装器 `to_tfp`。经过包装的 Distrax 分布可以直接在 TFP 中使用：

```python
key = jax.random.PRNGKey(1234)

distrax_dist = distrax.Normal(0., 1.)
wrapped_dist = distrax.to_tfp(distrax_dist)
metadist_tfp = tfd.Sample(wrapped_dist, sample_shape=[3])

samples = metadist_tfp.sample(seed=key)
print(metadist_tfp.log_prob(samples))  # 输出 -3.3409896
```

### 双射变换器 (Bijectors)

Distrax 中的“双射器（bijector）”是一个可逆函数，且知晓如何计算其雅可比行列式对数（log Jacobian determinant）。双射器可用于通过变换简单分布来构建复杂的概率分布。Distrax 双射器在功能上与 TFP 双射器类似，仅有少许 API 差异。以下是两者对比的示例：

```python
import distrax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

# 等效的变换分布
distrax.Transformed(distrax.Normal(loc=0., scale=1.), distrax.Tanh())
tfd.TransformedDistribution(tfd.Normal(loc=0., scale=1.), tfb.Tanh())
```

此外，Distrax 双射器支持复合与求逆：

```python
bij_distrax = distrax.Tanh()
bij_tfp = tfb.Tanh()

# 等效的双射器
inv_bij_distrax = distrax.Inverse(bij_distrax)
inv_bij_tfp = tfb.Invert(bij_tfp)

# 两者均为恒等双射器
distrax.Chain([bij_distrax, inv_bij_distrax])
tfb.Chain([bij_tfp, inv_bij_tfp])
```

所有 TFP 双射器都可以直接传递给 Distrax，并可以与 Distrax 双射器自由组合。例如以下用法均受支持：

```python
distrax.Inverse(tfb.Tanh())

distrax.Chain([tfb.Tanh(), distrax.Tanh()])

distrax.Transformed(tfd.Normal(loc=0., scale=1.), tfb.Tanh())
```

Distrax 双射器亦可传递给 TFP，但首先需要使用 `to_tfp` 进行转换：

```python
bij_distrax = distrax.to_tfp(distrax.Tanh())

tfb.Invert(bij_distrax)

tfb.Chain([tfb.Tanh(), bij_distrax])

tfd.TransformedDistribution(tfd.Normal(loc=0., scale=1.), bij_distrax)
```

Distrax 还提供了 `Lambda`，这是一个便捷的包装器，可将简易 JAX 函数快速转换为双射器。以下是若干 `Lambda` 示例及其 TFP 等效实现：

```python
distrax.Lambda(lambda x: x)
# tfb.Identity()

distrax.Lambda(lambda x: 2*x + 3)
# tfb.Chain([tfb.Shift(3), tfb.Scale(2)])

distrax.Lambda(jnp.sinh)
# tfb.Sinh()

distrax.Lambda(lambda x: jnp.sinh(2*x + 3))
# tfb.Chain([tfb.Sinh(), tfb.Shift(3), tfb.Scale(2)])
```

与 TFP 不同，Distrax 中的双射器在计算雅可比行列式时不接收 `event_ndims` 参数。相反，Distrax 假定每个双射器的事件维度数量在静态时是已知的，并使用 `Block` 将双射器提升到不同的维度。例如：

```python
x = jnp.zeros([2, 3, 4])

# 在 TFP 中，`event_ndims` 可以传递给双射器
bij_tfp = tfb.Tanh()
ld_1 = bij_tfp.forward_log_det_jacobian(x, event_ndims=0)  # Shape = [2, 3, 4]

# Distrax 默认假定 `Tanh` 是标量双射器
bij_distrax = distrax.Tanh()
ld_2 = bij_distrax.forward_log_det_jacobian(x)  # ld_1 == ld_2

# 设置 `event_ndims=2` 时，TFP 对 log det 的后两个维度求和
ld_3 = bij_tfp.forward_log_det_jacobian(x, event_ndims=2)  # Shape = [2]

# Distrax 静态处理维度数量
bij_distrax = distrax.Block(bij_distrax, ndims=2)
ld_4 = bij_distrax.forward_log_det_jacobian(x)  # ld_3 == ld_4
```

Distrax 双射器实现了 `forward_and_log_det` 方法（部分双射器还额外实现了 `inverse_and_log_det`），允许在一行代码中同时获取正向映射输出及其对数雅可比行列式。对于某些双射器而言，这比分别调用 `forward` 和 `forward_log_det_jacobian` 更加高效（同理，在可用时，`inverse_and_log_det` 比分别调用 `inverse` 和 `inverse_log_det_jacobian` 更高效）。

```python
x = jnp.zeros([2, 3, 4])
bij_distrax = distrax.Tanh()

y = bij_distrax.forward(x)
ld = bij_distrax.forward_log_det_jacobian(x)

# 上述操作的单行等效形式：
y, ld = bij_distrax.forward_and_log_det(x)
```

### JIT 编译支持 (Jitting Distrax)

Distrax 分布和双射器可以直接作为参数传递给经 JIT 编译的函数。用户自定义的分布和双射器只需分别继承 `distrax.Distribution` 和 `distrax.Bijector`，即可免费获得该特性。例如：

```python
mu_0 = jnp.array([-1., 0., 1.])
sigma_0 = jnp.array([0.1, 0.2, 0.3])
dist_0 = distrax.MultivariateNormalDiag(mu_0, sigma_0)

mu_1 = jnp.array([1., 2., 3.])
sigma_1 = jnp.array([0.2, 0.3, 0.4])
dist_1 = distrax.MultivariateNormalDiag(mu_1, sigma_1)

jitted_kl = jax.jit(lambda d_0, d_1: d_0.kl_divergence(d_1))

# 两者均输出 85.237
print(jitted_kl(dist_0, dist_1))
print(dist_0.kl_divergence(dist_1))
```

##### 关于 `vmap` 和 `pmap` 的说明

使 Distrax 对象能够作为参数传递给 JIT 编译函数的序列化逻辑，同样允许函数使用 `jax.vmap` 和 `jax.pmap` 将它们作为数据进行映射。

然而，***对该特性的支持目前是实验性且不完全的。对接收 Distrax 对象作为输入参数或返回 Distrax 对象的函数应用 `jax.vmap` 或 `jax.pmap` 时，请务必保持谨慎。***

简单对象（如 `distrax.Categorical`）在此类变换下可能表现正常，但较复杂的对象（如 `distrax.MultivariateNormalDiag`）用作被 `vmap` 或 `pmap` 装饰函数的输入或输出时可能会产生异常。

### 继承与扩展分布与双射器 (Subclassing Distributions and Bijectors)

用户自定义的概率分布可以通过继承 `distrax.Distribution` 来创建。只需实现少数几个方法即可完成：

```python
class MyDistribution(distrax.Distribution):

  def __init__(self, ...):
    ...

  def _sample_n(self, key, n):
    samples = ...
    return samples

  def log_prob(self, value):
    log_prob = ...
    return log_prob

  def event_shape(self):
    event_shape = ...
    return event_shape

  def _sample_n_and_log_prob(self, key, n):
    # 可选方法。仅在存在更高效实现时重写。
    samples, log_prob = ...
    return samples, log_prob
```

类似地，更复杂的双射器可以通过继承 `distrax.Bijector` 来创建。通常只需实现一至两个类方法：

```python
class MyBijector(distrax.Bijector):

  def __init__(self, ...):
    super().__init__(...)

  def forward_and_log_det(self, x):
    y = ...
    logdet = ...
    return y, logdet

  def inverse_and_log_det(self, y):
    # 可选方法。若无需逆向方法可省略。
    x = ...
    logdet = ...
    return x, logdet
```

## 示例程序 (Examples)

`examples` 目录包含若干使用 Distrax 的完整代表性程序：

- `hmm.py`：展示了如何使用 `distrax.HMM` 组合对隐马尔可夫模型（HMM）的初始状态、状态转移和观测分布进行建模的分布，并推断变化噪声信号中的潜在速率与状态转移。
- `vae.py`：包含变分自编码器（VAE）的示例实现，该模型通过像素上的联合 `distrax.Bernoulli` 分布对二值化 MNIST 数据集进行训练建模。
- `flow.py`：展示了使用 `distrax.MaskedCoupling` 层构建标准化流（Normalizing Flow）对 MNIST 数据建模并通过梯度下降进行训练的简洁示例。

## 致谢 (Acknowledgements)

我们衷心感谢 TensorFlow Probability 作者们在协助 Distrax 设计与跨框架兼容性方面提供的持续支持。

特别鸣谢 Aleyna Kara 与 Kevin Murphy 贡献了隐马尔可夫模型及其关联示例所基于的代码。

## 引用 Distrax (Citing Distrax)

本仓库是 DeepMind JAX 生态系统的一部分。如需引用 Distrax，请使用以下 BibTeX 条目：

```bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/deepmind},
  year = {2020},
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月23日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
