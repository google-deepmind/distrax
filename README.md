# Distrax

![CI status](https://github.com/deepmind/distrax/workflows/tests/badge.svg)
![pypi](https://img.shields.io/pypi/v/distrax)

Distrax is a lightweight library of probability distributions and bijectors. It
acts as a JAX-native reimplementation of a subset of
[TensorFlow Probability](https://www.tensorflow.org/probability) (TFP), with
some new features and emphasis on extensibility.

## Installation

You can install the latest released version of Distrax from PyPI via:

```sh
pip install distrax
```

or you can install the latest development version from GitHub:

```sh
pip install git+https://github.com/deepmind/distrax.git
```

To run the tests or
[examples](https://github.com/deepmind/distrax/tree/master/examples) you will
need to install additional [requirements](https://github.com/deepmind/distrax/tree/master/requirements).

## Design Principles

The general design principles for the DeepMind JAX Ecosystem are addressed in
[this blog](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research).
Additionally, Distrax places emphasis on the following:

1. **Readability.** Distrax implementations are intended to be self-contained
and read as close to the underlying math as possible.
2. **Extensibility.** We have made it as simple as possible for users to define
their own distribution or bijector. This is useful for example in reinforcement
learning, where users may wish to define custom behavior for probabilistic agent
policies.
3. **Compatibility.** Distrax is not intended as a replacement for TFP, and TFP
contains many advanced features that we do not intend to replicate. To this end,
we have made the APIs for distributions and bijectors as cross-compatible as
possible, and provide utilities for transforming between equivalent Distrax and
TFP classes.

## Features

### Distributions

Distributions in Distrax are simple to define and use, particularly if you're
used to TFP. Let's compare the two side-by-side:

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

# Both print 1.775
print(dist_distrax.log_prob(samples))
print(dist_tfp.log_prob(samples))
```

In addition to behaving consistently, Distrax distributions and TFP
distributions are cross-compatible. For example:

```python
mu_0 = jnp.array([-1., 0., 1.])
sigma_0 = jnp.array([0.1, 0.2, 0.3])
dist_distrax = distrax.MultivariateNormalDiag(mu_0, sigma_0)

mu_1 = jnp.array([1., 2., 3.])
sigma_1 = jnp.array([0.2, 0.3, 0.4])
dist_tfp = tfd.MultivariateNormalDiag(mu_1, sigma_1)

# Both print 85.237
print(dist_distrax.kl_divergence(dist_tfp))
print(tfd.kl_divergence(dist_distrax, dist_tfp))
```

Distrax distributions implement the method `sample_and_log_prob`, which provides
samples and their log-probability in one line. For some distributions, this is
more efficient than calling separately `sample` and `log_prob`:

```python
mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.1, 0.2, 0.3])
dist_distrax = distrax.MultivariateNormalDiag(mu, sigma)

samples = dist_distrax.sample(seed=key, sample_shape=())
log_prob = dist_distrax.log_prob(samples)

# A one-line equivalent of the above is:
samples, log_prob = dist_distrax.sample_and_log_prob(seed=key, sample_shape=())
```

TFP distributions can be passed to Distrax meta-distributions as inputs. For
example:

```python
key = jax.random.PRNGKey(1234)

mu = jnp.array([-1., 0., 1.])
sigma = jnp.array([0.2, 0.3, 0.4])
dist_tfp = tfd.Normal(mu, sigma)

metadist_distrax = distrax.Independent(dist_tfp, reinterpreted_batch_ndims=1)
samples = metadist_distrax.sample(seed=key)
print(metadist_distrax.log_prob(samples))  # Prints 0.38871175
```

To use Distrax distributions in TFP meta-distributions, Distrax provides the
wrapper `to_tfp`. A wrapped Distrax distribution can be directly used in TFP:

```python
key = jax.random.PRNGKey(1234)

distrax_dist = distrax.Normal(0., 1.)
wrapped_dist = distrax.to_tfp(distrax_dist)
metadist_tfp = tfd.Sample(wrapped_dist, sample_shape=[3])

samples = metadist_tfp.sample(seed=key)
print(metadist_tfp.log_prob(samples))  # Prints -3.3409896
```

### Bijectors

A "bijector" in Distrax is an invertible function that knows how to compute its
Jacobian determinant. Bijectors can be used to create complex distributions by
transforming simpler ones. Distrax bijectors are functionally similar to TFP
bijectors, with a few API differences. Here is an example comparing the two:

```python
import distrax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

# Same distribution.
distrax.Transformed(distrax.Normal(loc=0., scale=1.), distrax.Tanh())
tfd.TransformedDistribution(tfd.Normal(loc=0., scale=1.), tfb.Tanh())
```

Additionally, Distrax bijectors can be composed and inverted:

```python
bij_distrax = distrax.Tanh()
bij_tfp = tfb.Tanh()

# Same bijector.
inv_bij_distrax = distrax.Inverse(bij_distrax)
inv_bij_tfp = tfb.Invert(bij_tfp)

# These are both the identity bijector.
distrax.Chain([bij_distrax, inv_bij_distrax])
tfb.Chain([bij_tfp, inv_bij_tfp])
```

All TFP bijectors can be passed to Distrax, and can be freely composed with
Distrax bijectors. For example, all of the following will work:

```python
distrax.Inverse(tfb.Tanh())

distrax.Chain([tfb.Tanh(), distrax.Tanh()])

distrax.Transformed(tfd.Normal(loc=0., scale=1.), tfb.Tanh())
```

Distrax bijectors can also be passed to TFP, but first they must be transformed
with `to_tfp`:

```python
bij_distrax = distrax.to_tfp(distrax.Tanh())

tfb.Invert(bij_distrax)

tfb.Chain([tfb.Tanh(), bij_distrax])

tfd.TransformedDistribution(tfd.Normal(loc=0., scale=1.), bij_distrax)
```

Distrax also comes with `Lambda`, a convenient wrapper for turning simple JAX
functions into bijectors. Here are a few `Lambda` examples with their TFP
equivalents:

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

Unlike TFP, bijectors in Distrax do not take `event_ndims` as an argument when
they compute the Jacobian determinant. Instead, Distrax assumes that the number
of event dimensions is statically known to every bijector, and uses
`Block` to lift bijectors to a different number of dimensions. For example:

```python
x = jnp.zeros([2, 3, 4])

# In TFP, `event_ndims` can be passed to the bijector.
bij_tfp = tfb.Tanh()
ld_1 = bij_tfp.forward_log_det_jacobian(x, event_ndims=0)  # Shape = [2, 3, 4]

# Distrax assumes `Tanh` is a scalar bijector by default.
bij_distrax = distrax.Tanh()
ld_2 = bij_distrax.forward_log_det_jacobian(x)  # ld_1 == ld_2

# With `event_ndims=2`, TFP sums the last 2 dimensions of the log det.
ld_3 = bij_tfp.forward_log_det_jacobian(x, event_ndims=2)  # Shape = [2]

# Distrax treats the number of dimensions statically.
bij_distrax = distrax.Block(bij_distrax, ndims=2)
ld_4 = bij_distrax.forward_log_det_jacobian(x)  # ld_3 == ld_4
```

Distrax bijectors implement the method `forward_and_log_det` (some bijectors
additionally implement `inverse_and_log_det`), which allows to obtain the
`forward` mapping and its log Jacobian determinant in one line. For some
bijectors, this is more efficient than calling separately `forward` and
`forward_log_det_jacobian`. (Analogously, when available, `inverse_and_log_det`
can be more efficient than `inverse` and `inverse_log_det_jacobian`.)

```python
x = jnp.zeros([2, 3, 4])
bij_distrax = distrax.Tanh()

y = bij_distrax.forward(x)
ld = bij_distrax.forward_log_det_jacobian(x)

# A one-line equivalent of the above is:
y, ld = bij_distrax.forward_and_log_det(x)
```

### Jitting Distrax

Distrax distributions and bijectors can be passed as arguments to jitted
functions. User-defined distributions and bijectors get this property for free
by subclassing `distrax.Distribution` and `distrax.Bijector` respectively. For
example:

```python
mu_0 = jnp.array([-1., 0., 1.])
sigma_0 = jnp.array([0.1, 0.2, 0.3])
dist_0 = distrax.MultivariateNormalDiag(mu_0, sigma_0)

mu_1 = jnp.array([1., 2., 3.])
sigma_1 = jnp.array([0.2, 0.3, 0.4])
dist_1 = distrax.MultivariateNormalDiag(mu_1, sigma_1)

jitted_kl = jax.jit(lambda d_0, d_1: d_0.kl_divergence(d_1))

# Both print 85.237
print(jitted_kl(dist_0, dist_1))
print(dist_0.kl_divergence(dist_1))
```

##### A note about `vmap` and `pmap`

The serialization logic that enables Distrax objects to be passed as arguments
to jitted functions also enables functions to map over them as data using
`jax.vmap` and `jax.pmap`.

However, ***support for this behavior is experimental and incomplete. Use
caution when applying `jax.vmap` or `jax.pmap` to functions that take Distrax
objects as arguments, or return Distrax objects.***

Simple objects such as `distrax.Categorical` may behave correctly under these
transformations, but more complex objects such as
`distrax.MultivariateNormalDiag` may generate exceptions when used as inputs or
outputs of a `vmap`-ed or `pmap`-ed function.


### Subclassing Distributions and Bijectors

User-defined distributions can be created by subclassing `distrax.Distribution`.
This can be achieved by implementing only a few methods:

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
    # Optional. Only when more efficient implementation is possible.
    samples, log_prob = ...
    return samples, log_prob
```

Similarly, more complicated bijectors can be created by subclassing
`distrax.Bijector`. This can be achieved by implementing only one or two class
methods:

```python
class MyBijector(distrax.Bijector):

  def __init__(self, ...):
    super().__init__(...)

  def forward_and_log_det(self, x):
    y = ...
    logdet = ...
    return y, logdet

  def inverse_and_log_det(self, y):
    # Optional. Can be omitted if inverse methods are not needed.
    x = ...
    logdet = ...
    return x, logdet
```

## Examples

The `examples` directory contains some representative examples of full programs
that use Distrax.

`hmm.py` demonstrates how to use `distrax.HMM` to combine distributions that
model the initial states, transitions, and observation distributions of a
Hidden Markov Model, and infer the latent rates and state transitions in a
changing noisy signal.

`vae.py` contains an example implementation of a variational auto-encoder that
is trained to model the binarized MNIST dataset as a joint `distrax.Bernoulli`
distribution over the pixels.

`flow.py` illustrates a simple example of modelling MNIST data using
`distrax.MaskedCoupling` layers to implement a normalizing flow, and training
the model with gradient descent.

## Acknowledgements

We greatly appreciate the ongoing support of the TensorFlow Probability authors
in assisting with the design and cross-compatibility of Distrax.

Special thanks to Aleyna Kara and Kevin Murphy for contributing the code upon
which the Hidden Markov Model and associated example are based.

## Citing Distrax

This repository is part of the [DeepMind JAX Ecosystem]. To cite Distrax please
use the [DeepMind JAX Ecosystem citation].

[DeepMind JAX Ecosystem]: https://deepmind.com/blog/article/using-jax-to-accelerate-our-research "DeepMind JAX Ecosystem"
[DeepMind JAX Ecosystem citation]: https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt "Citation"
