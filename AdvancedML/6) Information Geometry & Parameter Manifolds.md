
# Submanifolds
So far, we have worked with distance measures over manifolds by measuring curve lengths along the manifold. Can the *original data space*/*ambient space* have a *Riemannian Metric*?

## Composite Functions
Lets recall that *manifolds* are spanned by a function $f:\mathbb{R}^d\to \mathbb{R}^D$,
$$ \mathcal{M} = f(\Omega), \Omega \subseteq \mathbb{R}^d. $$

Let's now assume $f$ is a ==composite function== i.e.,
$$ f(\mathbf{x}) = f_{2}(f_{1}(\mathbf{x})), $$
where $f_{1}:\mathbb{R}^d\to \mathbb{R}^{d'}, f_{2}:\mathbb{R}^{d'}\to \mathbb{R}^D$, where $d\leq d'\leq D$. This could be a multi-layered NN. Thus far we have assumed that $\Omega \subseteq \mathbb{R}^D$ which then was equipped with the *Riemannian Metric* $\mathbf{G_{x}}=\mathbf{J}_{\mathbf{x}}^\top\mathbf{J_{x}}$. The same analysis should then allow for $\Omega'=f_{1}(\Omega)\subseteq \mathbb{R}^{d'}$ to have a *metric*:
$$ \mathbf{G}_{\mathbf{x}}^{(2)}=\mathbf{J}_{\mathbf{x}}^{(2)\top}\mathbf{J}_{\mathbf{x}}^{(2)}, $$
where $\mathbf{J}_{\mathbf{x}}^{(2)} = \frac{\partial f_{2}}{\partial f_{1}(f)}$. In this instance, $\mathbf{G}_{x}$ represents the Euclidean metric of $\mathbb{R}^D$ brought into $\mathbb{R}^d$, and $\mathbf{G}^{(2)}_{\mathbf{x}}$ as $\mathbb{R}^D$ brought into $\mathbb{R}^{d'}$. ==A question is then==, if we bring the metric $\mathbf{G}^{(2)}_{\mathbf{x}}$ from $\mathbb{R}^{d'}$ into $\mathbb{R}^d$, do we get $\mathbf{G}_{\mathbf{x}}$?

For the answer, consider the manifold
$$ \mathcal{M}_{1} = f_{1}(\Omega) \subset \mathbb{R}^{d'}. $$
We use the Riemannian metric $\mathbf{G}^{(2)}_{\mathbf{x}}$ to measure curve lengths in $\mathbb{R}^{d'}$
$$ \begin{align}
\text{Length}(c) &= \int_{0}^1\left|\left|\frac{\text{d}}{\text{d}t} f_{1}(c_{t}) \right|\right|_{\mathbf{G}^{(2)}_{\mathbf{x}}} \text{d}t \\
&= \int_{0}^1\left|\left| \mathbf{J}_{c_{t}}^{(1)}\dot{c}_{t} \right|\right|_{\mathbf{G}^{(2)}_{\mathbf{x}}} \text{d}t,
\end{align} $$
where $\mathbf{J}_{\mathbf{x}}^{(1)} = \frac{\partial f_{1}}{\partial \mathbf{x}}$ is the Jacobian of $f_{1}$. Inserting into the definition of [inner norm](4\)%20Representation%20invariance#Local%20Norms%20&%20Angles)
$$ \begin{align}
\text{Length}(c) &= \int_{0}^1\sqrt{ \left(\mathbf{J}_{\dot{c}_{t}}^{(1)}\right)^\top \mathbf{G}_{c_{t}}^{(2)}\left(\mathbf{J}_{\dot{c}_{t}}^{(1)}\right) }\text{d}t \\
&= \int_{0}^1\sqrt{ \dot{c}_{t}^\top \left(\mathbf{J}_{\dot{c}_{t}}^{(1)\top} \mathbf{G}_{c_{t}}^{(2)}\mathbf{J}_{\dot{c}_{t}}^{(1)} \right) \dot{c}_{t} } \text{d}t,
\end{align} $$
That is $\Omega$ is not equipped with the metric $\mathbf{J}_{c_{t}}^{(1)\top}\mathbf{G}_{c_{t}}^{(2)}\mathbf{J}_{c_{t}}^{(1)}$. That means we can write the Jacobian of $f$ in terms of the Jacobians of $f_1$ and $f_2$:
$$ \mathbf{G}_{\mathbf{x}} = \mathbf{J}_{\mathbf{x}}^\top\mathbf{J_{x}} = \mathbf{J}_{c_{t}}^{(1)}\mathbf{J}_{c_{t}}^{(2)\top}\mathbf{J}_{c_{t}}^{(2)}\mathbf{J}_{c_{t}}^{(1)} = \mathbf{J}_{c_{t}}^{(1)\top}\mathbf{G}_{c_{t}}^{(2)}\mathbf{J}_{c_{t}}^{(1)}. $$

So $J^{(1)}$ measures the first transformation (if I have this speed and direction in my first manifold, how does this translate to the intermediate submanifold $\mathbb{R}^{d'}$) and $J^{(2)}$ measure the second transformation (if I have this velocity in the intermediate submanifold, how does this translate to the geometry space $\mathbb{R}^D$).

# Statistical Manifolds
What if $f$ outputs probability distributions? This we call ==information geometry== then, and manifolds are ==statistical manifolds==.

## Space of Distributions
The idea is simple: consider distribution $p(y|\theta)$ that is governed by set of parameters $\theta$. The ==*space of distributions*== will then simply be the space in which $\theta$ resides, and is ==denoted with== $\mathcal{H}$.

For example, if we have the univariate Gaussian $\mathcal{N}(y|\mu,\sigma^2)$, then the parameters are $\theta=(\mu,\sigma)$. The Gaussian distribution parameters space is 2D with $\mathbb{R}\times \mathbb{R}_{+}$, since $\mu$ is real values, and $\sigma$ is real valued and positive.

![[GaussianDistribtutionParameter.png]]

Similarly for the multivariate, we have $\theta=(\mu,\Sigma)$, but in this case $\sigma\in \mathbb{R}^D$ and the covariance is $\Sigma\in\text{Sym}_{+}^{D\times D}$, and so $\mathcal{H}=\mathbb{R}^D\times\text{Sym}_{+}^{D\times D}$.

Lastly, consider *categorical* distributions over $K$ possible classes has parameters $\theta=(p_{1},\dots,p_{K})$, where they sum to 1. The space of distributions $\mathcal{H}$ is then the unit simplex (set of positive numbers summing to 1).

## Naive Distances between distributions
With the distribution space $H$, we seek a distance measure. The Euclidean distance is an obvious first try. For two univariate Gaussians, the distance becomes
$$ \text{dist}(p_{1},p_{2}) = \sqrt{ (\mu_{1}-\mu_{2})^2+(\sigma_{1}-\sigma_{2})^2 },$$
however, had we chosen to parametrize the normal distribution as $\theta=(\mu,\sigma^2)$, it would give another distance. We want a distance metrics that does not depend on the parametrization. A ==reparameterization invariance== distance measure.

### Divergences are not distances
Kullback-Leibler divergence is defined as
$$ \text{KL}(p||q) = \int \log\left( \frac{p(y)}{q(y)} \right)p(y)\text{d}y = \mathbb{E}_{p(y)}\left[ \log \frac{p(y)}{q(y)} \right], $$for two distributions. It is by construction invariant to reparameterizations as it works with the densities. It has some nice properties; only when $p(y)=q(y)$ do we have equality between distributions. However, it is *not symmetric*.

## Curve Lengths & Energies; revisited
We can use the KL divergence to arrive at a distance measure however in the space of distributions. Recall the length of a curve, but now in the space of distributions and sligthly modified
$$ \text{Length}(c) = \lim_{ N \to \infty } \sum_{i=0}^N\text{dist}(p(y|c(t_{i})),\text{ }p(y|c(t_{i+1}))). $$
Similarly the *energy* of a curve
$$ \mathcal{E}(c) = \lim_{ N \to \infty } \sum_{i=0}^N\text{dist}^2(p(y|c(t_{i})),\text{ }p(y|c(t_{i+1}))) $$
We see that the distance function should be only locally defined between the two points.

### The Fisher-Rao Metric
The can understand local behavior of KL divergence through Taylor expansion. Let $p(y)=p(y|\theta)$ be a distribution parametrized by $\theta$, and let $q(y)=p(y|\theta+\delta)$ be an infinitesimal displacement with respect to $\theta$. The ==second order expansion around== $\theta$
$$ \text{KL}(p||q) = -\frac{1}{2}\delta^\top\underbrace{\left(\int p(y)\nabla_{\theta}^2\log p(y)\text{d}y\right)}_{\mathbf{F}}\delta+\mathcal{O}\left(||\delta||^3\right), $$
where $\nabla_{\theta}^2$ denotes the Hessian. Infinitesimally, we see that KL-divergence is a simple quadratic equation $\text{KL}(p||q)=\delta^\top\mathbf{F}\delta$. The matrix $\mathbf{F}$ can be written more elegantly formed as 
$$ \begin{align}
\mathbf{F} & = -\frac{1}{2}\int p(y)\nabla^2\log p(y)\text{d}y \\
&= \frac{1}{2}\int p(y)\nabla_{\theta}\log p(y)\nabla_{\theta}\log p(y)^\top\text{d}t
\end{align} $$

That is, locally the KL divergence is a quadratic function given as the expectation of the outer product of the gradient of $\log p(y)$ by itself. By definition the matrix $\mathbf{F}$ is positive definite, and is thus a *Riemannian Metric* over the space of distributions! This is commonly called the ==Fisher-Rao Metric==!

Summary, KL divergence is not a distance metric, but *locally* it is. Even better, locally it is a *Riemannian Metric*.

## Natural Gradients
A gradient of some loss with respect to the parameters of a statistical model. Unlike traditional Euclidean gradients, this takes the geometry of the space of distributions into account.

We usually use this to minimize some loss function $L$ with respect to parameters. The simplest algorithm *Gradient Descent* takes a small step in the direction of the gradident
$$ \theta^{(t+1)} = \theta^{(t)} - \alpha \frac{\partial L}{\partial \theta^{(t)}}, $$
with step size $\alpha$. The gradient is the direction of steepest descent.

# Noisy Manifolds
As we have limited data, our manifolds are only estimated.

## Manifold Learning; revisited

We want a $d$-dimensional manifold fitted to data $y_{1:N}=\{y_{1}\dots,y_{N}\}\in \mathbb{R}^D$. For autoencoders, this is done by minimizing loss of projections:
$$ f^* = \arg\min_{f}\sum_{n=1}^N|| f(\mathbf{x}_{n})-y_{n} ||^2, $$
where $\mathbf{x}_{n}$ is chosen such that $f(\mathbf{x}_{n})$ is the projection of $y_{n}$ onto the manifold spanned by $f$
$$ \mathbf{x}_{n} = \text{proj}_{\mathcal{M}}(\mathbf{y}_{n})=\arg\min_{\mathbf{x}}|| f(\mathbf{x}) - \mathbf{y}_{n} ||^2.$$
Where $f$ is a NN or similar, and we optimize its weights. This manifold will be "smooth" due to regularization or other reasons, but this means the geodesics may fail to reflect the structure of the data. See figure below.
![[NoisyManifold.png]]
That is, we would not expect the geodesics to cross over in the middle!

### Gaussian Noisy Manifolds
The above is not surprising, given that we do not have a lot of data, and that we are trying to fit a 2D plane in 3D to some data. This implies that $f$ will be poorly estimated in regions near no data.

So we instead consider a ==probabilistic model== with additive Gaussian noise
$$ \mathbf{y} = f(\mathbf{x}) + \epsilon \sigma(\mathbf{x}),\tag{1}$$
where the noise $\epsilon\in \mathbb{R}^{D\times D}$ is a diagonal matrix with standard gaussian entries $[\epsilon]_{{ii}}\sim\mathcal{N}(0,1)$, and $\sigma(\mathbf{x})\in \mathbb{R}^D$ is a vector of standard deviations that may change smoothly with $\mathbf{x}$.
$$ p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|f(x),\text{diag}(\sigma^2(\mathbf{x}))), $$
which could be *variational autoencoders*!

## Randomly Projected Manifolds
Eq (1) above in Noisy Manifolds spans a *manifold plus some noise*. How does out geometric tools apply to this?

We will take a view that the *noisy manifold* is a ==random projection== of a deterministic manifold. Eq (1) is matrix form
$$ \begin{align}
\mathbf{y}&=f(\mathbf{x})+\epsilon \sigma(\mathbf{x}) \\
&= (\mathbf{I},\epsilon)\begin{pmatrix}
f(\mathbf{x}) \\
\sigma(\mathbf{x})
\end{pmatrix} = \mathbf{P}h(x),
\end{align} $$
where $\mathbf{P}=(\mathbf{I},\epsilon)\in \mathbb{R}^{D\times2D}$ is a *random matrix* (random entries), which projects from $2D$ to $D$, and thus we can think $h(\mathbf{x}) = (f(\mathbf{x});\sigma(\mathbf{x}))$ as spanning a $d$ dimensional manifold in $\mathbb{R}^{2D}$, which is randomly projected by $\mathbf{P}$ to form $\mathbf{y}$.

### Example
Consider the 2D points of a ring. Then embedded through non-linear means to $\mathbb{R}^{1000}$. So it forms a bent ring in high dimension. Guassian noise is added, and fit a randomly projected manifold

![[Randommanifold.png]]
If we consider $h(\mathbf{x})=(f(\mathbf{x});\sigma(\mathbf{x}))$ and its *Riemannian metric*
$$ \mathbf{G}_{\mathbf{x}} = \mathbf{J}_{f}^\top(\mathbf{x})\mathbf{J}_{f}(\mathbf{x}) + \mathbf{J}_{\sigma}^\top(\mathbf{x})\mathbf{J}_{\sigma}(\mathbf{x}),$$
where the noise is also considered, we get a much better geodesics (along the ring). In this example, $\sigma$ takes low values near data, but larger values when not, that guides the geodesics.

## Statistical Submanifolds

## Pulling Back Fisher-Rao

# Parameter Manifolds

Consider a predictive model given by the function $f_{\theta}$ that maps an observation into the parameters of the distribution $p(y|x)$. For example, a *classifier* maps the observation into a vector of class probabilities. We write
$$f_{\theta}:\mathbb{R}^d \mapsto \mathcal{H},$$
where $\mathcal{H}$ is the ==space of probability distributions==. The parameters $\theta \in \mathbb{R}^P$ are estimated from data through the log likelihood
$$\theta_{\text{opt}} = \arg\max_{\theta}\sum_{n=1}^N\log p(y_{n}|x_{n}) = \arg\max_{\theta}\sum_{n=1}^N\log p(y_{n}|f_{\theta}(x_{n})).$$

We want geometric structure in the parameter space of $\mathbb{R}^P$.
## Overparametrization
Simple example to illustrate how parameter space can take a non-trivial geometric structure. Define $f_{\theta}:\mathbb{R}\to \mathbb{R},$
$$ f_{\theta}(\mathbf{x})=\theta_{2}\text{ReLU}(\theta_{1}\mathbf{x}). $$
It is a piece-wise linear function. For any $\alpha>0$, it can be reparametrized as
$$f_{\theta}(\mathbf{x})=\theta_{2}\text{ReLU}(\theta_{1}\mathbf{x})=\frac{\theta_{2}}{\alpha}\text{ReLU}(\alpha \theta_{1}\mathbf{x}).$$
The parameters $\theta=(\theta_{1},\theta_{2})$ generate the same as $\theta'=\left(\alpha \theta_{1},\frac{\theta_{2}}{\alpha} \right)$. So we have the ==identifiability problem==. Illustrated below

![[Overparametrization.png]]
^paramspace

We want a distance measure such that
$$\text{Dist}(\theta,\theta')=0\text{ for }\theta=(\theta_{1},\theta_{2})\text{ and }\theta'=\left( \alpha \theta_{1},\frac{\theta_{2}}{\alpha} \right),$$
and clearly Euclidean geometry does not work.

## Function-space metrics
An option is to use a metric over the generated functions to give a parameter space a natural metric. This could be the *expected inner product* between function outputs with respect to density $p(\mathbf{x})$:
$$ \langle f_{\theta}, f_{\theta'}\rangle = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[f_{\theta}(\mathbf{x})^\top f_{\theta'}(\mathbf{x})]. \tag{2}$$

We then see that for infinitesimal inner products get that
$$ \begin{align}
\langle \delta_{1},\delta_{2}\rangle_{\theta} &= \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[(f_{\theta+\delta_{1}}(\mathbf{x})-f_{\theta}(\mathbf{x}))^\top( f_{\theta+\delta_{2}}(\mathbf{x}) - f_{\theta}(\mathbf{x}) )] \\
&= \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[(\mathbf{J}_{\theta}(\mathbf{x})\delta_{1})^\top(\mathbf{J}_{\theta}(\mathbf{x})\delta_{2})] \\
&= \delta_{1}\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[\mathbf{J}_{\theta}(\mathbf{x})^\top\mathbf{J}_{\theta}(\mathbf{x})]\delta_{2},
\end{align} $$
where the Jacobian is with respect to the parameters and depend on $\mathbf{x}$. The pullback metric is then
$$\mathbf{G}_{\theta} = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[\mathbf{J}_{\theta}(\mathbf{x})^\top\mathbf{J}_{\theta}(\mathbf{x})]. \tag{3}$$

This is constant when either function is reparametrized. It is not a Riemannian metric however

## Information Metrics
In Eq (2) we assumed an Euclidean metric over the output space. Since we only engage with the output-space inner product *locally*, we can choose any Riemannian metric $\mathbf{F}_{f_{\theta}(\mathbf{x})}$ for this space. We then get
$$ \langle \delta_{1},\delta_{2}\rangle_{\theta}^\mathbf{F} = \delta_{1}^\top\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[\mathbf{J}_{\theta}(\mathbf{x})^\top\mathbf{F}_{f_{\theta}(\mathbf{x})}\mathbf{J}_{\theta}(\mathbf{x})] \delta_{2},$$where a natural choice is the *Fisher-Rao Metric* we saw earlier! The parameter space metric becomes
$$ \mathbf{G}_{\theta} = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[\mathbf{J}_{\theta}(\mathbf{x})^\top\mathbf{F}_{f_{\theta}(\mathbf{x})}\mathbf{J}_{\theta}(\mathbf{x})]. \tag{4}$$

## Pseudo Metrics
This analysis works with the Function-Space Metric (3) or Information Metric (4) worked on above. Directions may exist in parameter space in which the associated function does *not change*, implying distances are zero in those directions. This means the metric $\mathbf{G}_{\theta}\in \mathbb{R}^{P\times P}$ has a ==null-space/kernel==. Let $R$ denote the rank of $\mathbf{G}_{\theta}$ such that the null space has dimensionality $P-R$. In practice, $R$ is not known analytically. The *Eigen decomposition of the metric*
$$ \begin{align}
\mathbf{G}_{\theta} &= \mathbf{V}_{\theta}\mathbf{\lambda}_{\theta}\mathbf{V}_{\theta}^\top \\
&= (\mathbf{V}_{\text{im}}\;\; \mathbf{V}_{\text{ker}})\begin{pmatrix}
\mathbf{\lambda}_{\text{im}} & \mathbf{0} \\
\mathbf{0} & \mathbf{0}
\end{pmatrix}(\mathbf{V}_{\text{im}}\;\;\mathbf{V}_{\text{ker}})^\top,
\end{align} $$
where $\mathbf{\lambda}_{\theta}$ is a diagonal matrix containing the eigenvalues of $\mathbf{G}_{\theta}$. As a *null space* exists, some eigenvalues are $0$! $\mathbf{V}_{ker}\in \mathbb{R}^{P\times P-R}$ is the orthonormal basis of the *kernel* (null space) and $\mathbf{V}_{ker}\in \mathbb{R}^{P\times R}$ is the orthonormal basis of the *image* of $\mathbf{G}_{\theta}$.

So the decomposition implies there are parts where locally nothing changes in the functions space (kernel) and where it does change (image).

### Kernel Manifolds
The metric kernel is the set of possible directions where in parameter space the function do not change locally. Assuming that $f_{\theta}$ is smooth, then so is the Jacobian $\mathbf{J}_{\theta}(\mathbf{x})$. The eigenvectors and values also then change smoothly. We can the trace out trajectories in parameter space with $0$ length! Formally, an ==improper curve== $c:[0,1]\to \mathbb{R}^P$ if its speed $||\dot{c}_{t}||_{\mathbf{G}_{c_{t}}}$ is $0$ for all $t$!

From a given startpoint $\theta\in \mathbb{R}^P$, we can consider the set of points along such proper curves that connect to the starting point. In the picture of the parameter space figure [here](#^paramspace), it corresponds to the orange curved lines.
$$ \mathcal{M}_{\text{ker}}^\theta = \left\{ \theta'\in \mathbb{R}^P \middle\vert \exists c\text{ an improper curve such that } c_{0}=\theta\text{ and }c_{1}=\theta' \right\} $$
All pair-wise distances are 0 along these paths.

### Image Manifolds
Similarly for curves with strictly positive speeds. We formalize it as ==proper curves==: $c:[0,1]\to \mathbb{R}^P$ if its speed $||\dot{c}_{t}||_{\mathbf{G}_{c_{t}}}\geq 0$ for all $t$.
$$  \mathcal{M}_{\text{im}}^\theta = \left\{ \theta'\in \mathbb{R}^P \middle\vert \exists c\text{ a proper curve such that } c_{0}=\theta\text{ and }c_{1}=\theta' \right\}  $$

The natural choice of Riemannian metric for this $R$-dimensional manifold is
$$ \mathbf{G}_{\theta}^\text{im} = \mathbf{V}_{\text{im}}\mathbf{\lambda}_{\text{im}}\mathbf{V}_{\text{im}}^\top = \mathbf{G}_{\theta} $$
![[ParamSpaceImage.png]]
We decomposed the metric into a kernel and image; these are *orthonormal* as you can see in the picture.

## Finite Sample Metric Estimators
So far considered metrics as expectations over a data distribution
$$ \mathbf{G}_{\theta} = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}[\mathbf{J}_{\theta}(\mathbf{x})^\top\mathbf{J}_{\theta}(\mathbf{x})]. $$

We rarely in practice have access to $p(\mathbf{x})$. And if we had, the closed form may not exist anyway. We would resort to *monte-carlo estimation* of the metric
$$ \bar{\mathbf{G}}_{\theta} = \frac{1}{S}\sum_{s=1}^{S} \mathbf{J}_{\theta}(\mathbf{x}_{s})^\top\mathbf{F}_{f_{\theta}(\mathbf{x}_{s})}\mathbf{J}_{\theta}(\mathbf{x}_{s}), \tag{5}$$
where the samples $\mathbf{x}_{s}$ are assumed to be drawn from $p(\mathbf{x})$. In practice samples from our training data. The above metric in (5) is the ==average metric==, is an instance of a ==generalized Gauss-Newton Matrix==.