
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

Amortized variant is very strong, as for a single model, it will also return the parameters of the distribution for the input.