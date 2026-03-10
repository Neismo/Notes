
# More on Covariance Functions
Recall: a ==covariance function== $k:\mathcal{X}\times \mathcal{X}\to \mathbb{R}$ maps a pair of inputs $\mathbf{x}_{1},\mathbf{x}_{2}\in \mathcal{X}$ from some input space to the real line, and it is given by
$$ \mathbf{K}_{ij} = \text{Cov}(f(\mathbf{x}_{i}),f(\mathbf{x}_{j})) = k(\mathbf{x}_{i},\mathbf{x}_{j}), $$
so it is *independent* of the function realizations $f(\circ)$, and only based on the input. The covariance function **must be** ==symmetric== and ==positive semi-definite==
$$ \begin{align}
\mathbf{K} &= \mathbf{K}^\top\tag{Symmetric} \\
\forall \mathbf{x}&\neq 0:\quad \mathbf{x}^\top\mathbf{K}\mathbf{x}\geq 0,\tag{PSD} 
\end{align} $$
Some are also **stationary** and **isotropic**. *Stationary* means we can write the covariance function as just a function of the *difference* in the input, $k(\mathbf{x}_{i},\mathbf{x}_{j}) = k(\mathbf{x}_{i}-\mathbf{x}_{j})$. It is *isotropic* when we can write it as the *norm* between the difference, $k(\mathbf{x}_{i},\mathbf{x}_{j}) = k(||\mathbf{x}_{i}-\mathbf{x}_{j}||)$. *Isotropic* implies *stationary*.

## Feature Expansion
For the linear model, from which we derived the covariance function, was $f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x})$; the choice of $\phi(\mathbf{x})$ is *crucial*.

**Example**: binary classification problem in 2D is ***not always linearly separable***; embedding in 3D can make it linearly separable if we choose $\phi(\mathbf{x})$ correctly; see image below for example, where the feature expansion is $\phi(\mathbf{x})=\left[ x_{1},x_{2},\sqrt{ x_{1}^2+x_{2}^2 } \right]$.

![[ClassificationFeatureExpansion.png]]

### Kernels & Feature Spaces
As the linear model is $f(\mathbf{x}) = \mathbf{w}^\top\phi(\mathbf{x})$ with *Gaussian* priors $p(\mathbf{w}|\mathbf{0},\alpha^{-1}\mathbf{I})$ yielded
$$ \mathbf{K}_{ij} = \text{cov}(f(\mathbf{x}_{i}),f(\mathbf{x}_{j})) = \alpha^{-1}\phi(\mathbf{x}_{i})^\top\phi(\mathbf{x}_{j}), $$
so changing the feature expansion $\phi(\mathbf{x})$ changes the covariance function - does changing the covariance function then also change the feature expansion? The answer is **yes**:

**Example**: consider the kernel simply being $k(\mathbf{x}_{i},\mathbf{x}_{j}) = (\mathbf{x}_{i}^\top\mathbf{x}_{j})^2$, for $\mathbf{x}=[x_{1},x_{2}]\in \mathbb{R}^2$. Then 
$$ \begin{align}
k(\mathbf{x}_{i},\mathbf{x}_{j}) &= (\mathbf{x}_{i}^\top\mathbf{x}_{j})^2 \\
&= (x_{i,1}x_{j,1}+x_{i,2}x_{j,2})^2 \\
&= x_{i,1}^2x_{j,1}^2 +x_{i,2}^2x_{j,2}^2+2x_{i,1}x_{j,1}x_{i,2}x_{j,2} \\
&= \begin{bmatrix}
x_{i,1}^2 \\
x_{i,2}^2 \\
\sqrt{ 2 }x_{i,1}x_{i,2}
\end{bmatrix}^\top \begin{bmatrix}
x_{j,1}^2 \\
x_{j,2}^2 \\
\sqrt{ 2 }x_{j,1}x_{j,2}
\end{bmatrix}
\end{align} $$
So this kernel embeds 2D points in 3D feature space, without us ever constructing a 3D feature space explicitly. For similar reasons one can show that the *squared exponential* covariance function is **infinitely dimensional**!

# Gaussian Process Classification

We want to not only classify, but also specify our certainty in the classification. We talk about *aleatoric* and *epistemic* uncertainty; irreducible and reducible uncertainties.

Bayesian model for logistic regression, where $y_{n}\in\{0,1\}$
$$ \begin{align}
y_{n} &\sim \text{Ber}(\sigma(f(\mathbf{x}_{n}))) \\
f(\mathbf{x}_{n})&=\phi(\mathbf{x}_{n})^\top\mathbf{w} \\
\mathbf{w}&\sim\mathcal{N}(\mathbf{0},\alpha^{-1}\mathbf{I})
\end{align} $$

in Gaussian Process classification
$$ \begin{align}
y_{n}&\sim \text{Ber}(\sigma(f(\mathbf{x}_{n}))) \\
f&\sim \mathcal{GP}(0, k(\mathbf{x},\mathbf{x}'))
\end{align} $$
The function $\sigma:\mathbb{R}\to(0,1)$ is called a ==inverse link function==
1) Sigmoid: $\sigma(x) = \frac{1}{1+\exp(-x)}$
2) CDF of standard normal Gaussian: $\Phi(x) = \int_{-\infty}^x\mathcal{N}(x|0,1)\text{d}x$
Sigmoid can be more robust, but CDF has nice properties.

Likelihood for logistic regression, where $f_{n}=f(\mathbf{x}_{n})$
$$ p(\mathbf{y}|\mathbf{f}) = \prod_{n=1}^N \sigma(f_{n})^{y_{n}}(1-\sigma(f_{n}))^{1-y_{n}} $$
Our Gaussian process prior
$$ p(\mathbf{f}) = \mathcal{N}(\mathbf{f}|\mathbf{0},\mathbf{K}) $$
The predictive distribution for the classification case is not easily computed in close form
$$ p(y^* = 1|\mathbf{y},\mathbf{x^*}) = \int p(y^*=1|f^*)p(f^*|\mathbf{y,x^*}) \text{d}f^* \tag{Pred Post}, $$
where $f^* = f(x^*)$
To solve this we will use ==Laplace== transformation in 2 steps
1) Approximate $p(f^*|\mathbf{y},\mathbf{x}^*)$ using Laplace
2) Compute $p(y^*|\mathbf{y},\mathbf{x}^*)$

If we go from (Pred Post) we can rewrite $p(f^*|\mathbf{y},\mathbf{x}^*)$
$$ \begin{align}
 p(f^*|\mathbf{y},\mathbf{x}^*) &= \int p(f^*,\mathbf{f}|\mathbf{y},\mathbf{x}^*) \text{d}\mathbf{f} \tag{Sum rule} \\
 &=\int p(f^*|\mathbf{f, y, x}^*)p(\mathbf{f}|\mathbf{y,x}^*)\text{d}\mathbf{f}\tag{Product Rule} \\
 &= \int p(f^*|\mathbf{f, x}^*)p(\mathbf{f}|\mathbf{y})\text{d}\mathbf{f}\tag{Cond. Independence} \\
 &\approx \int p(f^*|\mathbf{f, x}^*)q(\mathbf{f})\text{d}\mathbf{f}\tag{Laplace}
\end{align}$$
where $q(\mathbf{f}) = \mathcal{N}(\mathbf{f}|\mathbf{m},\mathbf{S})$ is the Laplace approximation for $p(\mathbf{f}|\mathbf{y})$, where $\mathbf{m}_{i}$ is the approximate posterior mean of $f(\mathbf{x}_{i})$ and similarly for the variance $\mathbf{S}_{ii}$.
We also recall that $p(f^*|\mathbf{f}\mathbf{x}^*)$ is a conditional Gaussian, for which we know the results, just stated here:
$$ p(f^*|\mathbf{f},\mathbf{x}^*) = \mathcal{N}(f^*|\mathbf{k}^\top\mathbf{K^{-1}\mathbf{f}}, k-\mathbf{k}^\top\mathbf{K^{-1}\mathbf{k}}), $$
where $\mathbf{k}$ is the covariance between new points and training points $k(x^*,x_{n})$ and $\mathbf{K}$ is the covariance between all training points $k(x_{n},x_{m})$.

This means we again have two Gaussian in the integral and we form a closed form solution again
$$ \begin{align}
p(f^*|\mathbf{y},\mathbf{x}^*) &\approx \int\mathcal{N}(f^*|\mathbf{k}^\top\mathbf{K^{-1}\mathbf{f}}, k-\mathbf{k}^\top\mathbf{K^{-1}\mathbf{k}})\mathcal{N}(\mathbf{f}|\mathbf{m},\mathbf{S}) \\
&= \mathcal{N}(f^*|\mu_{f^*},\sigma^2_{f^*}) \\
&\mu_{f^*} = \mathbf{k^\top}\mathbf{K^{-1}}\mathbf{m} \\
&\sigma^2_{f^*} = k - \mathbf{k}^\top\mathbf{K}^{-1}(\mathbf{K-S})\mathbf{K}^{-1}\mathbf{k}
\end{align} $$
## Computational steps
>[!note] Classification Steps
>1) Compute the Laplace transformation
>$$ p(\mathbf{f}|\mathbf{y}) \approx q(\mathbf{f}) = \mathcal{\mathbf{f}|\mathbf{m},\mathbf{S}} $$
>2) Posterior distribution for latent function $f^*$ using the Laplace approximation
>$$ p(f^*|\mathbf{y},\mathbf{x}^*) = \mathcal{N}(f^*|\mu_{f^*},\sigma^2_{f^*}) $$
>3) Options for computing the predictive distribution for classification labels:
>$$ p(y^*=1|\mathbf{y},\mathbf{x}^*) \approx \int p(y^*=1|f^*)p(f^*|\mathbf{y},\mathbf{x}^*)\text{d}f^*=\int \sigma(f^*)p(f^*|\mathbf{y},\mathbf{x}^*)\text{d}f^* $$
>	1) Monte Carlo Sampling
>	$$ p(y^*=1|\mathbf{y},\mathbf{x}^*) \approx \frac{1}{S}\sum_{i=1}^S \sigma(f^{(i)}),\quad f^{(i)}\sim\mathcal{N}(f|\mu_{f^*},\sigma^2_{f^*}) $$
>	2) Probit Approximation
>	$$ \sigma(f) \approx \Phi\left( f\sqrt{ \frac{\pi}{8} } \right) \implies p(y^*=1|\mathbf{y},\mathbf{x}^*) \approx \Phi\left( \frac{\mu_{f^*}}{\sqrt{ \frac{8}{\pi} + \sigma^2_{f^*} }} \right) $$

