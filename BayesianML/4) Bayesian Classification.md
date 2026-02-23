
# Bayesian vs Classical Statistics


|                                 | **Frequentist / Classical**                   | **Bayesian**                                  |
| ------------------------------- | --------------------------------------------- | --------------------------------------------- |
| **Interpretation**              | Long run frequencies                          | Degrees of belief                             |
| **Parameters**                  | Deterministic, but unknown<br>Point Estimates | Random variables<br>Probability Distributions |
| **Interpretation of Intervals** | *Confidence Intervals*                        | *Credibility Intervals*                       |
| **Sources of Information**      | Data only                                     | Data + Prior knowledge                        |
| **Computation**                 | Often less expensive                          | Often more expensive                          |

# Bayesian Method for Classification

Starting with ==binary classification==:
- Dataset $\mathcal{D}\{ (\mathbf{x}_{1},y_{1}),\cdots,(\mathbf{x}_{N},y_{N}) \}$
	- Input features $\mathbf{x}_{i}\in \mathbb{R}^D$
	- Labels $y_{i}\in\{0,1\}$

We want to compute *predictive distributions*: what is the *probability* that my new point $\mathbf{x}^*$ belongs to class $1$? $p(y^*=1|\mathcal{D},\mathbf{x}^*)$? Two approaches
1) Discriminative Approach
2) Generative Approach

## Discriminative Vs Generative

The ==generative== approach models the *join distribution* $p(\mathbf{x}_{n},y_{n})$ via Bayes' rule:
$$ p(\mathbf{x}_{n},y_{n}) = \frac{p(\mathbf{x}_{n}|y_{n}=k)p(y_{n}=k)}{p(\mathbf{x}_{n})} $$
We can then generate new samples from the model:
$$ \mathbf{x}^{(i)},y^{(i)} \sim p(\mathbf{x},y)$$
This has some pros and cons:
- It is **optimal** if we get assumptions correct (specifically the likelihood can be problematic)
- Can easily handle **missing data**
- Can **reason about** input data ($p(\mathbf{x})$)
- Assumptions are **hard to get right** though!

The ==discriminative== approach models a *conditional distribution*, $p(y_{n}|\mathbf{x}_{n})$, assuming parametric form for posterior:
$$ p(y_{n}|\mathbf{x}_{n}) = f(\mathbf{x}_{n}|\mathbf{w}), $$
where $f(\circ)$ can be based on a linear model, neural network etc. This has some pros and cons as well:
- Often superior when the assumptions on the **generative model is wrong**
- Often **better calibrated**
- Easy to make **flexible**
- Difficult with **missing data**
- **Cannot reason** about input data

### The Generative Approach

Binary classification with $y_{n}\in\{0,1\}$:
$$  p(y_{n}=k|\mathbf{x}_{n}) = \frac{p(\mathbf{x}_{n}|y_{n}=k)p(y_{n}=k)}{p(\mathbf{x}_{n})},\tag{1} $$
- **Class-conditional probability**: $p(\mathbf{x}_{n}|y_{n}=k)$
- **Prior probability**: $p(y_{n}=k) = \pi_{k}$
- **Marginal Data Density**: $p(\mathbf{x}_{n})$

The marginal density can be found using *sum rule*:
$$ p(\mathbf{x}_{n}) = \sum_{k\in\{0,1\}}p(\mathbf{x}_{n}|y_{n}=k)\pi_{k} = p(\mathbf{x}_{n}|y_{n}=0)\pi_{0}+p(\mathbf{x}_{n}|y_{n}=1)\pi_{1}$$

If we plug that into (1) above we get:

$$ \begin{align}
p(y_{n}=1|\mathbf{x}_{n}) &= \frac{p(\mathbf{x}_{n}|y_{n}=1)\pi_{1}}{p(\mathbf{x}_{n}|y_{n}=0)\pi_{0}+p(\mathbf{x}_{n}|y_{n}=1)\pi_{1}} \\
&= \frac{1}{1+\frac{p(\mathbf{x}_{n}|y_{n}=0)\pi_{0}}{p(\mathbf{x}_{n}|y_{n}=1)\pi_{1}}} \tag{2}
\end{align}, $$
where a division by the numerator was done. If we now define
$$ a=\ln \frac{p(\mathbf{x}_{n}|y_{n}=1)\pi_{1}}{p(\mathbf{x}_{n}|y_{n}=0)\pi_{0}},$$
which is the *opposite* of the denominator we see in (2). Then we can plug it in and get
$$ p(y_{n}=1|\mathbf{x}_{n}) = \frac{1}{1+\exp(-a)} = \sigma(a)\tag{Sigmoid} $$

Which is the *Sigmoid function*! Similar reasoning for ==multi-class== can be done with $K$ classes:
$$ a_{k} = \ln p(\mathbf{x}_{n}|y_{n}=k)p(y_{n}=k), $$
where with similarly done for the binary class we can write:
$$ p(y_{n}|\mathbf{x}_{n}) = \frac{p(\mathbf{x}_{n}|y_{n}=k)p(y_{n}=k)}{\sum_{i=1}^Kp(\mathbf{x}_{n}|y_{n}=k)p(y_{n}=k)} = \frac{\exp(a_{k})}{\sum_{j=1}^K\exp(a_{j})}, \tag{Softmax} $$

#### With Gaussian data
When data follows gaussians, the term $a$ will be "simple" as the logarithm of a fraction of Gaussians has a simple solutions, (different means, same variance):

$$ a = \ln \frac{\pi_{1}}{\pi_{0}} - \frac{\mu_{1}^2-\mu_{0}^2}{2\sigma^{2}}+\frac{\mu_{1}^2-\mu_{0}^2}{\sigma^2}x_{n} = w_{0}+w_{1}x_{n} $$

### The Discriminative Approach

So in the *generative approach*, priors were defined for $p(y_{n}=k)=\pi_{k}$ and a set of class-conditionals $p(\mathbf{x}_{n}|\mathbf{y}_{n}=k)$, applied Bayes rule and ended up with:
$$ a(\mathbf{x}_{n}) = \ln \frac{\pi_{1}p(\mathbf{x}_{n}|y_{n}=1)}{\pi_{2}p(\mathbf{x}_{n}|y_{n}=2)} = \frac{p(y_{n}=1|\mathbf{x}_{n})}{p(y_{n}=0|\mathbf{x}_{n})}, $$
and
$$ p(y_{n}=1|\mathbf{x}_{n}) = \frac{1}{1+\exp(-a)} = \sigma(a(\mathbf{x}_{n})).$$
So the assumptions gave rise to a *functional form* for $a(\mathbf{x}_{n})$, but in ==discriminative modelling==, we directly assume a functional form for $a(\mathbf{x}_{n})$. For logistic regression, we would have a parameter $w$ instead:

$$  p(y_{n}=1|\mathbf{x}_{n}) = \frac{1}{1+\exp(-a)} = \sigma(\phi(\mathbf{x}_{n})^\top\mathbf{w}), $$
^likelihood

where each observation is modeled with a Bernoulli distribution with probability $\sigma(\phi(\mathbf{x}_{n})^\top\mathbf{w})$. Then it can be estimated with a **MAP**, **Maximum Likelihood** or **Bayesian Inference** on the likelihood functions:

$$ p(\mathbf{y}|\mathbf{x},\mathbf{w}) = \prod_{n=1}^N\sigma(\phi(\mathbf{x}_{n})^\top\mathbf{w})^{y_{n}}\left( 1-\sigma(\phi(\mathbf{x}_{n})^\top\mathbf{w})\right)^{1-y_{n}} $$

A problem here with optimizers, when maximizing, when the data is linearly separable, the weights tends towards infinity! Adding a prior can fix this (even if it is weak).

# Bayesian Logistic Regression

The problem with Bayesian Logistic Regression, is we lose ability to compute the *posterior distribution* and *posterior predictive distribution*. We saw earlier how **grid approximations** could be used, but it suffers from *curse of dimensionality*. Today we will see how *Laplace* transformations can help approximate the posterior.

Recall that the *likelihood* for logistic regression is:
$$  p(\mathbf{y}|\mathbf{x},\mathbf{w}) = \prod_{n=1}^N\sigma(\phi(\mathbf{x}_{n})^\top\mathbf{w})^{y_{n}}\left( 1-\sigma(\phi(\mathbf{x}_{n})^\top\mathbf{w})\right)^{1-y_{n}}.  $$

and we now impose a *prior* on the weights (i.i.d):
$$ p(\mathbf{w}) = \mathcal{N}(\mathbf{w}|\mathbf{0}, \alpha^{-1}\mathbf{I}) = \prod_{i=1}^D\mathcal{N}(w_{i}|0,\alpha^{-1}), $$
and then the posterior follows from Bayes':
$$ p(\mathbf{w}|\mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{w})p(\mathbf{w})}{p(\mathbf{y})}. $$

If we now want to calculate the posterior mean:
$$ \mathbb{E}_{p(\mathbf{w}|\mathbf{y})}[\mathbf{w}] = \int \mathbf{w}p(\mathbf{w}|\mathbf{y})\partial \mathbf{w}=\frac{1}{p(\mathbf{y})}\int \mathbf{w}p(\mathbf{y}|\mathbf{w})p(\mathbf{w})\partial \mathbf{w}, $$
and we see that we can't ignore the $p(\mathbf{y})$ distribution any longer unfortunately, like with MAP and MLE. Also the integral is very hard and has no closed form. This problem also exists for many other interesting summaries we would do on the posterior.

**Bernstein von Mises theorem:** if we assume certain regularity conditions, then the posterior distributions will look more and more like a Gaussian, as $N$ increases.
*Idea*: approximate $p(\mathbf{w}|\mathbf{y})$ with a Gaussian.

# Laplace Transformations

**Laplace** is a transformation that is used to approximate intractable probability densities:

$$ p(\mathbf{w}|\mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{w})p(\mathbf{w})}{p(\mathbf{y})} = \frac{1}{Z}f(\mathbf{w})\approx\mathcal{N}(\mathbf{w}|\mathbf{m},\mathbf{S}),$$
where $Z$ is a normalization constant, and $f(\circ)$ is a function that resembles the numerator part. The theorem above states it is approximated by a Gaussian. Recall the log density of a Gaussian:

$$ \ln\mathcal{N}(\mathbf{w}|\mathbf{m},\mathbf{S}) = -\frac{D}{2}\ln(2\pi)-\frac{1}{2}\ln|\mathbf{S}|-\frac{1}{2}(\mathbf{w}-\mathbf{m})^\top S^{-1}(\mathbf{w}-\mathbf{m}), \tag{3}$$
which is quadratic in $\mathbf{w}$. We now make a second-order Taylor expansion of $f(\mathbf{w})$ around the **MAP** estimator $\mathbf{w}_{MAP}$:
$$ \ln f(\mathbf{w}) \sim \ln f(\mathbf{w}_{MAP}) - \frac{1}{2}(\mathbf{w}-\mathbf{m})^\top \mathbf{A}(\mathbf{w}-\mathbf{m}), \tag{4}$$
where $A = -\nabla \nabla \ln f(\mathbf{w})|_{\mathbf{w}=\mathbf{w}_{MAP}}$ is the Hessian at the mode (MAP). The first order term is missing because we are at the mode (so the derivative is 0). Compare (3) to (4) above, and notice that $\mathbf{m}=\mathbf{w}_{MAP}$ and $S=A^{-1}$. So the ==Laplace transformation== is approximated as:
$$ q(\mathbf{w})=\mathcal{N}(\mathbf{w}|\mathbf{w}_{MAP},\mathbf{A}^{-1}). $$

### Summarize of steps:
1) Locate the MODE of the posterior:
   $$ \mathbf{w}_{MAP} = \arg\max_{\mathbf{w}}p(\mathbf{w}|\mathbf{y})=\arg\max_{\mathbf{w}}p(\mathbf{y}|\mathbf{w})p(\mathbf{w}) $$
2) Evaluate the Hessian at $\mathbf{w}_{MAP}$
   $$ \mathbf{A} = -\nabla \nabla \ln p(\mathbf{y}|\mathbf{w})p(\mathbf{w})|_{\mathbf{w}=\mathbf{w}_{MAP}} $$
**Advantages**:
- Simple and well-understood
- Very fast to compute
- Gives good results for many problems

**Limitations**:
- Only applies to continuous parameters
- Gaussian (symmetric distribution, thin tails)
- Only captures local properties of $p(\mathbf{w}|\mathbf{y})$ near the MAP solution
- Does *not* work for hierarchical models in general

### Approximating the Marginal Likelihood

We can also approximate the marginal likelihood. Our second order expansion for $\ln f(\mathbf{w})$ is
$$ \ln f(\mathbf{w}) = \ln f(\mathbf{w}_{MAP}) - \frac{1}{2}(\mathbf{w}-\mathbf{w}_{MAP})\mathbf{A}(\mathbf{w}-\mathbf{w}_{MAP}), $$
By assumption from earlier we had
$$ p(\mathbf{w}|\mathbf{y}) = \frac{1}{Z}f(\mathbf{w})\Rightarrow Z=\int f(\mathbf{w})\partial \mathbf{w}, $$
and we plug in our approximation for $\ln f(\mathbf{w})$
$$ Z=\int f(\mathbf{w})\partial \mathbf{w} \approx f(\mathbf{w}_{MAP})\int \exp\left( -\frac{1}{2} (\mathbf{w}-\mathbf{w}_{MAP})^\top\mathbf{A}i_{5}(\mathbf{w}-\mathbf{w}_{MAP}) \right) \partial \mathbf{w} = f(\mathbf{w}_{MAP})\frac{(2\pi)^{D/2}}{|\mathbf{A}|^{1/2}}$$

Where in the last step we use the fact that it is on the Gaussian functional form, which we *know* integrates to the normalization constant, which is the right side. 

Using $f(\mathbf{w})=p(\mathbf{y}|\mathbf{w})p(\mathbf{w}),$ our approximation becomes:
$$ \ln p(\mathbf{y}) \approx \ln p(\mathbf{y}|\mathbf{w}_{MAP}) + \ln p(\mathbf{w}_{MAP}) + \frac{D}{2}\ln(2\pi) - \frac{1}{2}\ln|\mathbf{A}|, $$
which was gotten for substituting the $f(\mathbf{w})$ into the above for $Z$, and taking the logarithm.

# Posterior Predictive Distribution

For (binary) classification, we need the *predictive posterior distribution* for a new input $\mathbf{x}^*$. The likelihood for a input data point $\mathbf{x}^*$ can be recalled from [earlier](#^likelihood):
$$ p(y^*=1|\mathbf{w},\mathbf{x}^*) = \sigma(\phi(\mathbf{x}^*)^\top\mathbf{w})$$
We now use *sum rule* and *posterior uncertainty* into account:
$$
\begin{align}
p(y^*=1|\mathbf{y},\mathbf{x}^*) &= \int p(y^*=1|\mathbf{x}^*,\mathbf{w})p(\mathbf{w}|\mathbf{y})\partial \mathbf{w} \\
&\approx \int p(y^*=1|\mathbf{x}^*,\mathbf{w})q(\mathbf{w})\partial \mathbf{w} \tag{Laplace Approx} \\
&=\int \sigma(\phi(\mathbf{x}^*)^\top\mathbf{w})\mathcal{N}(\mathbf{w}|\mathbf{m},\mathbf{S})\partial \mathbf{w} \\
&= \int \sigma(f)\mathcal{N}(f|\mu,\sigma^2)\partial f, \tag{Change of Variable}
\end{align}
$$
where $\mu=\phi(\mathbf{x}^*)^\top\mathbf{m}$ and $\sigma^2=\phi(\mathbf{x}^*)^\top\mathbf{S}\phi(\mathbf{x}^*)$. This changes the integral from a high dimensional problem to a 1D problem! This was from last week also. However, *still no closed form solution*! General strategies:

1) Monte Carlo Sampling 
   $$ p(y^*=1|\mathbf{y},\mathbf{x}^*) \approx \sum_{i=1}^S \phi(f^{(i)}),\text{  for  } f^{(i)}\sim\mathcal{N}(f|\mu,\sigma^2) $$
2) Numerical Integration (*Gauss-Hermite Integration*) - efficient when one of the components in the integral is Gaussian
   $$ p(y^*=1|\mathbf{y},\mathbf{x}^*) \approx \frac{1}{\sqrt{ \pi }}\sum_{i=1}^S w_{i}h(\sqrt{ 2 }\sigma x_{i}+\mu) $$
3) Probit approximation
   $$ \sigma(y)\approx \Phi\left( y\sqrt{ \frac{\pi}{8} } \right), $$
   where $\Phi$ is the CDF of the standard normal. Substituting the above into the integral provides a closed-form solution, and the approximation is very close to the sigmoid function.