
# Introduction

As an appetizer, consider that when we want to generate an image, say a horse, we have some *factors* we generate it from; shape, size, background, color, silouhette. We then add details from those factors.

In mathematical terms, say we have high dimensional $\mathbf{x}\in\mathcal{X}^D$ and a **low dimensional latent variable** $\mathbf{z}\in\mathcal{Z}^M$ that we can call hidden factors in data. $\mathcal{Z}^M$ can be referred to as a ==low dimensional manifold==. Then the generative process is:
$$\begin{align}
\mathbf{z} &\sim p(\mathbf{z}) \\
\mathbf{x} &\sim p(\mathbf{x}|\mathbf{z})
\end{align}$$

We first sample $\mathbf{z}$, and then we sample the data sample $\mathbf{x}$ from the conditional distribution $p(\mathbf{x}|\mathbf{z})$. 

However, when training we only have access to $p(\mathbf{x})$ and require *marginalizing* $\mathbf{z}$ out. So we get that the function we optimize over is:
$$p(\mathbf{x})=\int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})$$
# Probabilistic Principal Component Analysis
- We consider continuous random variables only: $\mathbf{z}\in\mathbb{R}^M$ and $\mathbf{x}\in\mathbb{R}^D$.
- The distribution of $\mathbf{z}$ is the standard Gaussian: $\mathcal{N}(\mathbf{z}|0,\mathbf{I})$.
- The dependency between $\mathbf{x}$ and $\mathbf{z}$ is linear, and we assume a Gaussian additive noise:
 $$\mathbf{x}=\mathbf{Wz}+\mathbf{b}+\epsilon,$$
where $\epsilon\sim\mathcal{N}(\epsilon|0,\sigma^2\mathbf{I})$. With the properties of Gaussian distributions, we get that:
$$ p(\mathbf{x}|\mathbf{z})=\mathcal{N}\left( \mathbf{x}|\mathbf{Wz}+\mathbf{b}, \sigma^2\mathbf{I} \right), $$
which is known as the *pPCA* model.

The integral then explicitly can be calculated:
$$\begin{align}
p(\mathbf{x}) &= \int_z p(\mathbf{x}|\mathbf{z})p(\mathbf{z}) \\
  &= \int_z\mathcal{N}\left(\mathbf{x}|\mathbf{Wz}+\mathbf{b},\sigma^2\mathbf{I}\right)\mathcal{N}(\mathbf{z}|0,\mathbf{I}) \\
  &=\mathcal{N}(\mathbf{x}|\mathbf{b,WW^\top}+\sigma^2\mathbf{I}).
\end{align}$$

The last step is because the mean and covariance of $\mathbf{x}=\mathbf{Wz}+\mathbf{b}+\epsilon$ from above, we get it to be respectively $\mathbf{b}$ and $\mathbf{WW}^\top + \sigma^2\mathbf{I}$:
$$ E[x] = E[Wz+b+\epsilon] = E[Wz] + E[b] + E[\epsilon] = 0 + \mathbf{b} + 0 $$$$ Cov[x] = Cov(Wz) + Cov(\epsilon) = WIW^\top + \sigma²I $$

# Variational Auto Encoders: Variational Inference for Nonlinear Latent Variable Models

## The Model and the Objective

Let us take the integral above in a more general case where we cannot calculate it directly; then ==simplest solution is Monte Carlo Sampling==:

$$\begin{align}
p(\mathbf{x}) &= \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z}) \\
  &= \mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}[p(\mathbf{x}|\mathbf{z})] \\
  &\approx \frac{1}{K}\sum_kp(\mathbf{x}|\mathbf{z_k}),
\end{align}$$

Where we sample $\mathbf{z}_k\sim p(\mathbf{z})$. This is a relatively easy approach. However, we can easily get into the trap of the *curse of dimensionality*. The number of samples needed grow exponentially with $M$, the dimensionality size. 

Another approach is ==variational inference==. Consider a family of variational distributions parameterized by $\phi$: $\{ q_\phi(\mathbf{z}) \}_\phi$. For Gaussian distributions, it would be $\phi = \{\mu, \sigma^2\}$. We know the form of the distributions, and we can assume they assign non-zero probability density to all $z\in\mathcal{Z}^M$. Then the logarithm of the marginal distribution is approximated as follows:

$$\begin{align}
\ln p(\mathbf{x}) &= \ln\int_z p(\mathbf{x}|\mathbf{z})p(\mathbf{z}) \\
  &= \ln\int_z\frac{q_\phi(\mathbf{z})}{q_\phi(\mathbf{z})}p(\mathbf{x|\mathbf{z}})p(\mathbf{z}) \\
  &=\ln\mathbb{E}_{z\sim q_\phi(\mathbf{z})}\left[\frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q_\phi(\mathbf{z})}\right] \\
  &\geq \mathbb{E}_{z\sim q_\phi(\mathbf{z})}\ln\left[ \frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q_\phi(\mathbf{x})} \right] \\
  &= \mathbb{E}_{z\sim q_\phi(\mathbf{z})}[\ln p(\mathbf{x}|\mathbf{z}) + \ln p(\mathbf{z}) - \ln q_\phi(\mathbf{z})] \\
  &= \mathbb{E}_{z\sim q_\phi(\mathbf{z})}[\ln p(\mathbf{x}|\mathbf{z})] - \mathbb{E}_{z\sim q_\phi(\mathbf{z})}[\ln q_\phi(\mathbf{z}) - \ln p(\mathbf{z})]
\end{align}$$

In line 4, *Jensen's Inequality* is used. An ==amortized variant== exists which uses $q_\phi(\mathbf{z}|\mathbf{x})$ instead:
$$ \ln p(\mathbf{x}) \geq \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}[\ln p(\mathbf{x}|\mathbf{z})] - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}[\ln q_\phi(\mathbf{z|\mathbf{x}}) - p(\mathbf{z})] $$

Amortized variant is very strong, as for a single model, it will also return the parameters of the distribution for the input. As a result, a *stochacstic encoder* $q_\phi(\mathbf{x}|\mathbf{z})$ and *stochastic decoder* $p(\mathbf{x}|\mathbf{z})$. This is also known as **Variational Auto Encoder**. The lower bound of the *log-likelihood function* is called the *Evidence Lower Bound* **ELBO**, it is decomposed as:

$$ \underbrace{\mathbb{E}_{z\sim q_\phi(\mathbf{z})}[\ln p(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction Error}} - \underbrace{\mathbb{E}_{z\sim q_\phi(\mathbf{z})}[\ln q_\phi(\mathbf{z}) - \ln p(\mathbf{z})]}_{\text{Regularizer}} $$

Where the *reguralizor* part coincides with the **Kullback-Leibler Divergence** (**KL**). 

### ELBO - a problem

ELBO provides a *lower limit* to the log-likelihood function. This is seen from the complete derivation for ELBO (steps not shown, but at page 97):

$$ \ln p(\mathbf{x}) = \underbrace{\mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}[\ln p(\mathbf{x}|\mathbf{z})] + KL[q_\phi(\mathbf{z}|\mathbf{x})||p(\mathbf{z})]}_{\text{ELBO}} - \underbrace{KL[q_\phi(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x}))]}_{\geq 0} $$

The last part is simply a term expressing how well the *stochastic* variational encoder is to the *true* posterior! However, we don't know $p(z|x)$, so we can't evalute the term. We can evaluate the ELBO loss though! Therefore, the last term can be though as the ==gap between the ELBO and *true* log-likelihood==!

In essence, this means that even if we minimize ELBO, if the variational posterier is a bad approximation, we might still have a large gap to the true maximum

![[ELBO-problem.png]]

## Components of VAEs

A class of *amortized* variational encoders $\{q_\phi(\mathbf{z}|\mathbf{x}\}_\phi$ that approximate the *true* posterior $p(\mathbf{z}|\mathbf{x})$. They are seen as **stochastic encoders**. The likelihood can be seen as **stochastic decoders**: $p(\mathbf{x}|\mathbf{z})$. Lastly, the **marginal distribution** $p(\mathbf{z})$, or **prior**, is left. We are also bounded on the objective ELBO:
$$ \ln p(\mathbf{x}) \geq \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})}[\ln p(\mathbf{x}|\mathbf{z})] + KL[q_\phi(\mathbf{z}|\mathbf{x})||p(\mathbf{z})] $$
1. How we parameterize the distributions?
2. How do we calculate the expected values? Integrals are still present.

### Parameterization of Distributions

We are free to use ==any distributions we want==, as long as they fit the problem. For images, a *categorical distribution* would fit. In many cases, we can use a neural network, however! With a softmax, that becomes a categorical distribution!

For the *latent variable*, we usually consider it a vector of real values, so a Gaussian is usually chosen. This combines to the fact:
$$\begin{align}
q_\phi(\mathbf{z}|\mathbf{x})&= \mathcal{N}\left(\mathbf{z}|\mu_\phi(\mathbf{x}),\text{diag}\left[ \sigma_\phi^2(\mathbf{x}) \right] \right)\\
p(\mathbf{z}) &= \mathcal{N}(\mathbf{z}|\mathbf{0},\mathbf{I}) \\
\end{align}$$

Here we can have the neural network output $2M$ outputs, $M$ mean values, and $M$ values for variance $\sigma^2$! That means we consider only *diagonal* variance!

### Reparameterization Trick

There is some variation when we sample for the posterior. We can reduce this variance w.r.t ELBO if we do the reparameterization trick. For the prior above, we can instead de-compose:

$$\mathbf{z}= \mu + \sigma \cdot \epsilon, $$

where $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. When we differentiate this, the variation from the $\epsilon$ term vanishes! We also only need to sample once during training with SGD!

## Potential Problems with VAEs

- *Posterior Collapse*: this happens when the *decoder* just models $\mathbf{z}$.
- *Hole Problem*: when the aggregated prior $q_\phi(\mathbf{z}) = \frac{1}{N}\sum_n q_\phi(\mathbf{z}|\mathbf{x}_n)$ mismatches the real prior $p(\mathbf{z})$, then there are regions where the prior is high probability, but low for the aggregated (and reverse). This produces low-quality results.
- *Out-of-distribution Problem*: VAEs often fail to detect out-of-distribution examples. This means, we would expect low probability for out-of-distribution examples, but in practice, it is now like that (funnily enough)

