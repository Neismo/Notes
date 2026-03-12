# Introduction
We will be working with ==Riemann Manifolds==. A manifold is a collection of points in some high dimension that forms a smooth object. In a *latent variable model*, we model our data as
$$ p(\mathbf{y}) = \int p(\mathbf{y}|\mathbf{x})p(\mathbf{x}) \text{d}x, \tag{1}$$
where $\mathbf{x}$ is the **latent variable** containing the essence of our observations.

## Probabilistic PCA
Assume $D$-dimensional data $\mathbf{y}\in \mathbb{R}^D$, and latent variable $\mathbf{x}\in \mathbb{R}^d,d<D$. The ==probabilistic principal component analysis== models an *affine transformation* between $\mathbf{x}$ and $\mathbf{y}$

 $$ p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{Ax}+\mathbf{b},\sigma^2\mathbf{I}), $$
where $\mathbf{A}\in \mathbb{R}^{D\times d}$ and $\mathbf{b}\in \mathbb{R}^D$. More informally $\mathbf{y} = f(\mathbf{x}) = \mathbf{Ax} + \mathbf{b} + \text{noise}$. We can also assume that $\mathbf{x}=\mathcal{N}(\mathbf{x}|\mathbf{0},\mathbf{I})$, and we have a *fully specified model*! The **closed form solution for (1)** is then

$$ p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{b},\mathbf{AA}^\top + \sigma^2\mathbf{I}). $$

This model is commonly used for visualization, but it is not ==identifiable==: the parameters can be changed *without* changing the distribution. Assume we rotate with a ==rotation matrix== $\mathbf{R}$: $\hat{\mathbf{x}}=\mathbf{Rx}$, then $p(\hat{\mathbf{x}})=\mathcal{N}(\mathbf{x}|\mathbf{0},\mathbf{I})$ - it *still* follows a unit Gaussian. Let $\hat{\mathbf{A}} = \mathbf{AR}^\top$, then the new likelihood is:
$$\hat{p}(\mathbf{y}|\hat{\mathbf{x}}) = \mathcal{N}(\mathbf{y}|\hat{\mathbf{A}}\hat{\mathbf{x}}+\mathbf{b},\sigma^2\mathbf{I}).$$

This again has a closed form solution:
$$ p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{b},\mathbf{AR}^\top \mathbf{RA}^\top + \sigma^2\mathbf{I}) = \mathcal{N}(\mathbf{y}|\mathbf{b},\mathbf{AA}^\top + \sigma^2\mathbf{I}), $$
which is the *original model*! Here we used that $\mathbf{R^\top R}=\mathbf{I}$. The ==takeaway here== is that *any rotation of the latent variable* will not give rise to a new visualization!

# Making Maps

Maps from our earth are usually flat, but it poses a a fundamentally impossible challenge. Gauss stated
> Gaussian curvature of a surface is preserved by local isometries.

>[!note] **Isometry**
>A transformation that *preserves* distances

Since a paper and a sphere can't have the same curvature, it can't be preserved with a transformation. This carries out to smaller regions as well: as long as there is curvature, we will get *distorted version* in a flat version!

*Distortions* will be smaller, though, if we consider small maps at a time.

# Embedded & Immersed Manifolds

We have observed data $\mathbf{y}_{1},\cdots,\mathbf{y}_{N}\in \mathbb{R}^D$, called the ==ambient space/latent space==. It is possible to fit on a lower-dimensional fold $\mathcal{M}$ with this data, through a *mapping*:
$$ f:\mathbb{R}^d\rightarrow \mathbb{R}^D,\quad d<D $$
from a *low-dimensional manifold* to a higher dimensional observation space. Different functions of $f$ give rise to different manifolds. We call *f* the *decoder*.

## Manifold Learning
A suitable *loss function* is selected that reflect the *properties we seek*. Perhaps it is *least squares*
$$ \theta^*,\mathbf{x}_{1}^*,\cdots,\mathbf{x}_{N}^*=\arg\min\mathcal{L}(\theta,\mathbf{x}_{1},\cdots,\mathbf{x}_{N}) = \arg\min\sum_{n=1}^N|| f_{\theta}(\mathbf{x}_{n})-\mathbf{y}_{n} ||_{2}^2. \tag{2} $$
Two important remarks:
1) *f* is **parameterized** by a vector $\mathbf{\theta}$. For polynomials, it contain coefficients, for neural network it contain weights; not written explicitly
2) As we do not know $f$ before, we do not know $\mathbf{x}_{n}$ as well. We find those by minimizing (2).

Other loss functions exists, but they should rely on some similarity measure between $f(\mathbf{x}_{n})$ and $\mathbf{y}_{n}$. Just like with *probabilistic PCA*, assume an optimal manifold has been found, then a invertible transformation of the latent variable $h:\mathbb{R}^d\rightarrow \mathbb{R}^d$:
$$\begin{align}
\hat{f}(\hat{\mathbf{x}}_{n})&=f(h^{-1}(\hat{\mathbf{x}}_{n})) \\
\hat{\mathbf{x}}_{n}&=h(\mathbf{x}_{n}),
\end{align}$$
it is clear then that $\hat{f}(\hat{\mathbf{x}}_{n})=f(\mathbf{x_{n}})$. Again, a learned manifold does not unique set of parameters. It is the ==reparameterization issue== or ==identifiability issue==. This tells us not to rely too heavily on the plots of a PCA for example.

In practice, if we fit a model to data several times, we may see different outcomes. VAEs may fit differently. 

## Statistical Identifiability
>[!note] Identifiable Model (*identifiability issue*)
>A statistical model $p_{\theta}$ parameterized by $\theta$ is said to be *identifiable* if the mapping $\theta\rightarrow p_{\theta}$ is *injective* (1-to-1). In other words, changing parameters ($\theta$) does not change the model!

**Examples** of identifiable models are *normal distributions*; if we change either $\mu,\sigma^2$, we get a new distribution/new model.
A *Mixture of Gaussians* is **not identifiable**
$$ p_{\theta}(\mathbf{y}) = \frac{1}{2}\mathcal{N}(\mathbf{y}|\mu_{1},\sigma_{1}^2)+\frac{1}{2}\mathcal{N}(\mathbf{y}|\mu_{2},\sigma_{2}^2), $$
which can be seen simply by swapping $\mu_{1}$ with $\mu_{2}$ or swapping $\sigma_{1}$ with $\sigma_{2}$. As we saw, *probabilistic PCA* was not either. If we have $\theta=\{ \mathbf{b},\mathbf{A},\sigma \}$, then we saw that $\theta'=\{\mathbf{b},\mathbf{AR^\top},\sigma\}$ gave rise to a similar model.

## Manifolds
Most loss functions do not have closed form solutions, which can be made easy with iterative functions with derivatives. We must therefore assume $f$ is smooth. Consider such a function
$$ f:\Omega \rightarrow \mathbb{R}^D, $$
with $\Omega \subseteq \mathbb{R}^d$ which could be the entire domain or a subset of it. Then
>[!note] Manifold $\mathcal{M}$
>A manifold $\mathcal{M}$ is the *image* of $\Omega$ under $f$
>$$ \mathcal{M} = f(\Omega) $$

Since $f$ is smooth, we can locally approximate it by its ==tangent space==. According to Taylor's, the **Jacobian** provides a basis for this tangent space.

$$ \mathbf{J}_{f} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_D}{\partial x_1} & \cdots & \frac{\partial fx_D}{\partial x_D} \end{bmatrix}. $$

### Embedded Manifolds
>[!note] **Embedded** manifolds
>A manifold is considered **embedded** if $f$ is **invertible** on $\mathcal{M}$. This means it locally does not self-intersect or locally change dimensionality. Below is an example of a *not* embedded manifold.
>![[NonEmbeddedManifold.png]]
>
>**Requirement**: if *f* is *injective*: $f(\mathbf{x}_{1})=f(\mathbf{x}_{2})\rightarrow \mathbf{x}_{1}=\mathbf{x}_{2}$ implies it is invertible, and thus also an embedded manifold.
>
>>[!note] **Immersed** manifolds
>>Sometimes it is *too restrictive* to work with embedded manifolds. We can require *local invertible* ones though: here we **require** the *Jacobian* $\mathbf{J}=\frac{\partial f}{\partial \mathbf{x}}$ to be of *full rank* for all $\mathbf{x}\in \Omega$.
>
>**Embedded** implies **immersed**, but not necessarily the other way around. Mostly work is done with *immersed* manifolds, as they are less restrictive.
^embedded
# Distances on Manifolds
In statistical models, we require a way to measure similarity. Introducing *Riemannian Geometry*.

## Euclidean Curve Lengths
Distances on manifolds in a basic idea is defined as the length of shortest paths. Let $c:[0,1]\rightarrow \mathbb{R}^D$ be a smooth curve in a *D*-dimensional space. We can the approximate the curve by connecting a set of straight lines from points on this curve
$$ \{(c(0),c(t_{1}))\},\{(c(t_{1}),c(t_{2}))\},\cdots,\{(c(t_{N}),c(1))\}, $$
where $t_{n}\in]0,1[, n=1\cdots N$. The sum of these segments then is the length of the curve
$$ \text{Length}(c)\approx\sum_{i=0}^N|| c(t_{i})-c(t_{i+1}) ||_{2}, $$
where $t_{0}=0,t_{N+1}=1$. Increasing the number of line segments increases the resolution/accuracy of course. The limit as $N\rightarrow \infty$ is the true length. Does so makes the sum to an integral and we get
$$ \begin{align}
\text{Length}(c) &= \lim_{ N \to \infty } \sum_{i=0}^N \left|\left|\frac{(c(t_{i})-c(t_{i+1}))}{\Delta t}\right|\right|\Delta t  \\
&= \int_{0}^1\left|\left| \frac{\partial c(t)}{\partial t} \right|\right|\text{d}t,
\end{align}, $$
where the integral is now over the velocity vectors!

## Reparametrization issues

A recurring problem also here is the ==reparametrization issue==; if we introduce a smooth monotonic function $h:[0,1]\rightarrow[0,1]$ with the properties
$$ h(0)=0,\quad h(1)=1, $$
we can then parameterize the curve $c$ by $\hat{t}=h(s)$. We then see that $\hat{c}(s)=c(h(s))$ generates the same curve as $c(s)$, but at a different speed. This is illustrated below:

![[ReparametrizationissueCurve.png]]

The lengths are also equal:
$$ \begin{align}
\text{Length}(c)&=\int_{0}^1\left|\left| \frac{\partial c}{\partial t} \right|\right|_{2} \text{d}t \\
&= \int_{0}^1\left|\left| \left( \frac{\partial c}{\partial t} \right)_{t=h(s)} \right|\right|_{2} \frac{\partial h}{\partial s} \text{d}s \tag{$t \rightarrow h(s)$} \\
&=\int_{0}^1\left|\left| \left( \frac{\partial c}{\partial t} \right)_{t=h(s)}\frac{\partial h}{\partial s} \right|\right|_{2} \text{d}s \tag{$\partial h/\partial s > 0$} \\
&=\int_{0}^1\left|\left| \frac{\partial c}{\partial s} \right|\right|_{2} \text{d}s \tag{Chain Rule} \\
&= \text{Length}(\hat{c})
\end{align} $$

This may be a problem because gradient-based optimization may land in a local optima which is not a unique solution to the curve.

### Measuring on Immersed Manifolds
Repeating on curves restricted to a manifold $\mathcal{M}=f(\Omega)$, where $\Omega$ is a connected subset of $\mathbb{R}^d$, such that the manifold is [immersed](#^embedded) in $\mathbb{R}^D$. In short, we will ==define the length of a latent curve to be the length of the *decoded curve*==.

So let $c:[0,1]\rightarrow \Omega$ be a smooth curve in $\Omega$ such that $f(c)$ is a smooth curve on $\mathcal{M}$. The curve length is then
$$ \text{Length}_{\mathcal{M}}(c)=\text{Length}(f(c))=\int_{0}^1\left|\left| \frac{\partial}{\partial t} f(c_{t}) \right|\right|_{2} \text{d}t, $$
where $c_{t}=c(t)$. Given two points $c_0,c_1\in\Omega$, we can define the shortest connecting path as
$$ \begin{align}
c_{01}&=\arg\min_{c}\text{Length}_{\mathcal{M}}(c) \\
& \text{subject to }c(0)=c_{0}\text{ and }c(1)=c_{1} 
\end{align} $$

We call above local minimizer a **geodesic**. Given the shortest connecting curve between two points, we define the **geodesic distance** as the length of this curve:
$$ \text{dist}(c_{0},c_{1}) = \text{Length}_{\mathcal{M}}(c_{01}). $$

This is a point along the *manifold*, so different from Euclidean.

As it is a proper distance metric it must satisfy the following
$$ \begin{align}
\text{dist}(x,y)&=0\Leftrightarrow x=y \\
\text{dist}(x,y)&=\text{dist}(y,x)  \\
\text{dist}(x,y)&\leq \text{dist}(x,z)+\text{dist}(z,y)
\end{align} $$

>[!faq] Distance summary
>Measure curve lengths in observation space, and define distances as the length of shortest paths.

## Reparametrization issue
The distance above - or the distance between two points - is *invariant to reparametrizations*. As usual
$$\begin{align}
\Omega &\subseteq \mathbb{R}^d \\
f&:\Omega\rightarrow \mathbb{R}^D,\quad d<D
\end{align}$$
such that the manifold $\mathcal{M}=f(\Omega)$. Consider the smooth invertible function $h:\mathbb{R}^d\rightarrow \mathbb{R}^d$, define
$$ \begin{align}
\hat{\Omega} &= h(\Omega)\subseteq \mathbb{R}^d \\
\hat{f}(\mathbf{x}) &= f(h^{-1}(\mathbf{x})),
\end{align} $$
then
$$ \begin{align}
\hat{f}(\hat{\Omega}) &= f(h^{-1}(h(\Omega))) \\
&= f(\Omega) \\
&= \mathcal{M}
\end{align} $$

As we can see, $\hat{\Omega},\hat{f}$ define the exact same manifolds as $\Omega,f$.

While data recovered from some algorithm may not be identifiable (suffer from reparametrization issue), the pairwise distances between recovered $\mathbf{x}_{1:N}$ can be made identifiable!

# Riemannian metrics
We can get a better intrinsic understanding of the length of a curve by applying chain rule
$$\begin{align}
\text{Length}_{\mathcal{M}}(c)&=\int_{0}^1\left|\left| \frac{\partial}{\partial t}f(c_{t}) \right|\right|_{2} \text{d}t \\
&= \int_{0}^1|| \mathbf{J}_{c_{t}}\dot{c}_{t} ||_{2}\text{d}t \\
&= \int_{0}^1\sqrt{ \dot{c}_{t}^\top\mathbf{J}_{c_{t}}^\top\mathbf{J}_{c_{t}}\dot{c}_{t} }\text{d}t, 
\end{align}$$
where $\dot{c}_{t}=\frac{\partial c}{\partial t}|_{t=t}$ is the ==*curve velocity*== and $\mathbf{J}_{c_{t}} = \frac{\partial f}{\partial x}|_{x=c_{t}}$ is the ==curvature== or **Jacobian** of $f$ evaluated at $c_{t}$.

## Tangent Spaces
The Jacobian can be seen as a local linear approximation of $f$. The Taylor's Theorem tells us
$$f(x+\epsilon)\approx f(x)+\mathbf{J}_{x}\epsilon,$$
for small $\epsilon$. Dimensions involved
$$\begin{align}
f &: \mathbb{R}^d\rightarrow \mathbb{R}^D \\
\mathbf{J}_{\mathbf{x}}&\in \mathbb{R}^{D\times d} \\
\mathbf{x,\epsilon}&\in \mathbb{R}^d \\
\mathbf{y} = f(\mathbf{x}) &\in \mathbb{R}^D.
\end{align}$$

Here $\mathbf{J}$ forms a a basis of a *d*-dimensional hyperplane in $\mathbb{R}^D$ that is **tangential to** $f(x)$. It is the ==tangent space== at $\mathbf{x}$ and **denoted as** $T_{\mathbf{x}}$. As was defined for [immersive manifolds](#^embedded), the Jacobian must have ==full rank everywhere==, and should have dimension $d$ at all points on the manifold.

## Local Inner Products
Local euclidean views are valuable, and was already made us of:
$$ \text{Length}_{\mathcal{M}}(c) = \int_{0}^1||\mathbf{J}_{c_{t}}\dot{c}_{t}||\text{d}t=\int_{0}^1||\mathbf{J}_{c_{t}}\underbrace{(\dot{c}_{t}\text{ d}t)}_{\text{Heuristic}}||, $$
which means we can think $\dot{c}_{t} \text{d}t$ as an infinitesmal vector which is then ==expressed in the tangent space==, where we measure it. The inner product between two such infinitesimal vectors, denoted $\delta_{1},\delta_{2}$, are displaced from a point $\mathbf{x}$. If we map both vectors to the tangent space at $\mathbf{x}$ and then compute the usual Euclidean inner product
$$ \langle\delta_{1},\delta_{2}\rangle_{\mathbf{x}}=(\mathbf{J}_{\mathbf{x}}\delta_1)^\top(\mathbf{J}_{\mathbf{x}}\delta_{2}) = \delta_{1}^\top\mathbf{J}_{\mathbf{x}}^\top\mathbf{J}_{\mathbf{x}}\delta_{2}, $$
where the subscript $\langle\circ\rangle_{\mathbf{x}}$ signifies it is calculated for a point $\mathbf{x}$ in the latent space.

## Local Norms & Angles

Given a vector $\mathbf{v}$ in the tangent space at $\mathbf{x}$, we can measure its norm using local inner product
$$ ||\mathbf{v}||_{\mathbf{x}} =\sqrt{ \langle\mathbf{v},\mathbf{v}\rangle_{\mathbf{x}} } = ||\mathbf{Jv}||_{2} = \sqrt{ \mathbf{v}^\top\mathbf{J}_{x}^\top\mathbf{J}_{x}\mathbf{v}  }$$

If two curves $c_{1}$ and $c_{2}$ intersect on the manifold at some point $\mathbf{x}$, we can get an *angle* between them. Commonly between two vectors $\mathbf{v}_{1},\mathbf{v}_{2}$
$$ \cos(\theta) = \frac{\langle\mathbf{v}_{1},\mathbf{v}_{2}\rangle}{||\mathbf{v}_{1}||_{2}\cdot||\mathbf{v}_{2}||_{2}}, $$
where $\theta$ is the angle of interest. If the vectors are now the curse velocities of $c_{1},c_{2}$ instead, then the angle is the angle between the velocities at the point of intersection. Formally, let $t_1,t_2$ be defined such that 
$$ c_{1}(t_{1}) = \mathbf{x} = c_{2}(t_{2}). $$
The velocities at the intersection point is then
$$\begin{align}
\mathbf{v}_{1} &= \dot{c}_{1}(t_{1}) \\
\mathbf{v}_{2} & =\dot{c}_{2}(t_{2}),
\end{align}$$
where $\dot{c}=\frac{\partial d}{\partial t}$ was short-hand for the curve derivatives (velocities). The angle now becomes
$$ \begin{align}
\theta &= \cos^{-1}\left( \frac{\langle\mathbf{v}_{1},\mathbf{v}_{2}\rangle_{\mathbf{x}}}{||\mathbf{v}_{1}||_{\mathbf{x}}\cdot||\mathbf{v}_{2}||_{\mathbf{x}}} \right) \\
&=\cos^{-1}\left( \frac{\mathbf{v}_{1}^\top\mathbf{J}_{\mathbf{x}}^\top\mathbf{J}_{\mathbf{x}}\mathbf{v}_{2}}{||\mathbf{J}_{\mathbf{x}}\mathbf{v}_{1}||\cdot||\mathbf{J}_{\mathbf{x}}\mathbf{v}_{2}||} \right)
\end{align} $$

The norms and angles are *reliable*! They are locally euclidean in the *data space*, but along the *manifold* in that space spanned by *f*.

## Riemannian Metrics
The Jacobian is of great interest, but it often appears as $\mathbf{J}^\top\mathbf{J}$. As such, it is given it's own name, ==metric matrix (Riemannian metric)==
$$ \mathbf{G}_{\mathbf{x}}=\mathbf{J}_{\mathbf{x}}^\top\mathbf{J}_{\mathbf{x}}, $$
where the subscript as usual denotes evaluation at $\mathbf{x}\in \Omega$. Inner products, for example, are now written as
$$ \langle \delta_{1},\delta_{2} \rangle_{\mathbf{x}} = \delta_{1}^\top\mathbf{G}_{\mathbf{x}}\delta_{2} $$
This means unit circles may form as ellipses, as the *metric matrix* might make steps in some directions longer or shorter!
![[InnerProductRiemannian.png]]

### Property 1: positive definiteness
Since $\mathbf{J}\in \mathbb{R}^{D\times d}$, then $\mathbf{G}\in \mathbb{R}^{d\times d}$ is a symmetric matrix. When the local manifold is **immersed**, then the Jacobian matrix has *full rank* and $\mathbf{G}$ is positive definite - all eigenvalues are strictly positive - which means a SVD decomposition exists for $\mathbf{J}$.

Applying this to $\mathbf{G}$ shows that the inner product is **positive definite** which means distances are strictly positive! This assumes the manifold is *immersive* though.

### Property 2: basis independence
$\mathbf{G}$ is independent of the rotation of $\mathbf{J}$ we have; if we rotate it, we still get $\mathbf{G}$. Say $\tilde{\mathbf{J}}=\mathbf{RJ}$, then
$$ \mathbf{\tilde{\mathbf{G}}}=\tilde{\mathbf{J}}^\top \tilde{\mathbf{J}} = (\mathbf{RJ})^\top(\mathbf{RJ}) = \mathbf{J}^\top\mathbf{R}^\top\mathbf{RJ} = \mathbf{J}^\top\mathbf{J} = \mathbf{G}. $$

## Riemannian Metrics & Manifolds

The metric function can be seen as $\mathbf{G}_{x}:\Omega\rightarrow \text{Sym}_{+}^{d\times d}$.

If the metric is a *smooth function*, then we call the manifold $\mathcal{M}$ as *Riemannian Manifold*. We want it smooth to have ==access to derivatives==.

## Measuring on Abstract Manifolds
With the metric, we can define
$$ \begin{align}
\langle \delta_{1},\delta_{2}\rangle_{\mathbf{x}} &= \delta_{1}^\top\mathbf{G}_{x}\delta_{2}\tag{Inner Product} \\
||\mathbf{v}||_{\mathbf{x}} &= \sqrt{ \mathbf{v}^\top\mathbf{G_{x}}\mathbf{v} }\tag{Norm} \\
\theta&=\cos^{-1}\left( \frac{\mathbf{v}_{1}^\top\mathbf{G_{x}}\mathbf{v}_{2}}{\sqrt{ \mathbf{v_{1}}^\top\mathbf{G_{x}}\mathbf{v_{1}}} \cdot \sqrt{ \mathbf{v_{2}}^\top\mathbf{G_{x}}\mathbf{v_{2}}} } \right)\tag{Angle} \\
\text{Length}_{\mathcal{M}}(c)&=\int_{0}^1 \sqrt{ \dot{c}_{t}^\top\mathbf{G}_{c_{t}}\dot{c}_{t} }\text{d}t, \tag{Curve Length}
\end{align} $$
# Exercises

### 1) Norm and Angle preserved

With $\hat{\mathbf{x}} = \mathbf{Rx}$, show that Euclidean distance between points in these representations are identical, i.e., $||\mathbf{x}_{i}-\mathbf{x}_{j}|| = || \hat{\mathbf{x}}_{i} - \hat{\mathbf{x}}_{j} ||$. Then show that the angle is also preserved i.e., $\angle(\mathbf{x}_{i}-\mathbf{x}_{j},\mathbf{x}_{k}-\mathbf{x}_{j}) = \angle(\hat{\mathbf{x}}_{i}-\hat{\mathbf{x}}_{j},\hat{\mathbf{x}}_{k}-\hat{\mathbf{x}}_{j})$.

$$ \begin{align}
|| \hat{\mathbf{x}}_{i} - \hat{\mathbf{x}}_{j} ||^2 &= (\mathbf{Rx}_{i}-\mathbf{Rx}_{j})^\top(\mathbf{Rx}_{i}-\mathbf{Rx}_{j}) \\
&= (\mathbf{R}(\mathbf{x}_{i}-\mathbf{x}_{j}))^\top(\mathbf{R}(\mathbf{x}_{i}-\mathbf{x}_{j})) \\
&= (\mathbf{x}_{i}-\mathbf{x}_{j})^\top\mathbf{R}^\top\mathbf{R}(\mathbf{x}_{i}-\mathbf{x}_{j}) \\
&= (\mathbf{x}_{i}-\mathbf{x}_{j})^\top(\mathbf{x}_{i}-\mathbf{x}_{j}) = ||\mathbf{x}_{i}-\mathbf{x}_{j}||^2
\end{align} $$
Above derivation show the squared distance are equal, so are the norms. Let us go to angles:
$$ \begin{align}
\angle(\hat{\mathbf{x}}_{i}-\hat{\mathbf{x}}_{j},\hat{\mathbf{x}}_{k}-\hat{\mathbf{x}}_{j}) &= \cos^{-1}\left( \frac{\langle\mathbf{\hat{x}}_{i}-\hat{\mathbf{x}}_{j},\mathbf{\hat{x}}_{k}-\hat{\mathbf{x}}_{j}\rangle}{||\mathbf{\hat{x}}_{i}-\hat{\mathbf{x}}_{j}||\cdot||\mathbf{\hat{x}}_{k}-\hat{\mathbf{x}}_{j}||} \right) \\
&= \cos^{-1}\left( \frac{(\mathbf{R}(\mathbf{x}_{i}-\mathbf{x}_{j}))^\top\mathbf{R}(\mathbf{x}_{k}-\mathbf{x}_{j})}{||\mathbf{x}_{i}-\mathbf{x}_{j}||\cdot||\mathbf{x}_{k}-\mathbf{x}_{j}||} \right) \\
&= \cos^{-1}\left( \frac{(\mathbf{x}_{i}-\mathbf{x}_{j})^\top\mathbf{R}^\top\mathbf{R}(\mathbf{x}_{k}-\mathbf{x}_{j})}{||\mathbf{x}_{i}-\mathbf{x}_{j}||\cdot||\mathbf{x}_{k}-\mathbf{x}_{j}||} \right) \\
&= \cos^{-1}\left( \frac{(\mathbf{x}_{i}-\mathbf{x}_{j})^\top(\mathbf{x}_{k}-\mathbf{x}_{j})}{||\mathbf{x}_{i}-\mathbf{x}_{j}||\cdot||\mathbf{x}_{k}-\mathbf{x}_{j}||} \right) \\
&= \angle(\mathbf{x}_{i}-\mathbf{x}_{j},\mathbf{x}_{k}-\mathbf{x}_{j}),
\end{align}  $$
where I used the fact from the previous part that $||\mathbf{x}_{i}-\mathbf{x}_{j}|| = || \hat{\mathbf{x}}_{i} - \hat{\mathbf{x}}_{j} ||$

### 2) Jacobian and immersive & embedded manifold
Consider the generative model $\mathbf{y} = f(\mathbf{x})$, where $\mathbf{x}=(x_{1},x_{2})\in \mathbb{R}^2$ and $\mathbf{y}=(y_{1},y_{2},y_{3})\in \mathbb{R}^3$. Let
$$ f(\mathbf{x})=\begin{pmatrix}
f_{1}(\mathbf{x}) \\
f_{2}(\mathbf{x}) \\
f_{3}(\mathbf{x})
\end{pmatrix} = \begin{pmatrix}
2x_{1}^2+x_{2}^2 \\
x_{1} \\
x_{2}
\end{pmatrix},$$
$$ \mathbf{J}_{f} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_D}{\partial x_1} & \cdots & \frac{\partial fx_D}{\partial x_D} \end{bmatrix}. $$
1) Derive the Jacobian matrix of $f$.

$$ \mathbf{J}_{f} = \begin{bmatrix}
4x_{1} & 2x_{2} \\
1 & 0 \\
0 & 1
\end{bmatrix}.$$
2. Show that the generative model spans an **immersed** manifold

$$ \text{Rank}(\mathbf{J}_{f}) = 2,$$
is required for it to be immersed. Let us calculate the row echelon form.
$$\text{ReducedForm}(\mathbf{J}_{f}) = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
4x_{1}-4x_{1} & 2x_{2}-2x_{2}
\end{bmatrix} = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 0
\end{bmatrix}$$
where I have swapped row 1 and row 2, then row 2 and row 3, and the $\text{row}_{3} = \text{row}_{3} - \text{row}_{1}\cdot4x_{1}-\text{row}_{2}\cdot_{2}x_{2}$. So it is an *immersive manifold*!

3) Show that the generative models spans an **embedded manifold**
For this we must show that $\mathbf{J}_{f}$ is invertible (left inverse):
$$ J_{f}^{-1} = (\mathbf{J}_{f}^\top\mathbf{J}_{f})^{-1}\mathbf{J}_{f}^\top, $$
so let us carry out the steps:
$$ \begin{align}
\mathbf{J}_{f}^\top\mathbf{J}_{f} &= \begin{bmatrix}
4x_{1} & 1 & 0 \\
2x_{2} & 0 & 1
\end{bmatrix}\begin{bmatrix}
4x_{1} & 2x_{2} \\
1 & 0 \\
0 & 1
\end{bmatrix} = \begin{bmatrix}
16x_{1}^2+1 & 8x_{1}x_{2} \\
8x_{1}x_{2} & 4x_{2}^2+1
\end{bmatrix} \\
(\mathbf{J}_{f}^\top\mathbf{J}_{f})^{-1} = \frac{1}{ad-bc}\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix} &= \frac{1}{64x_{1}^2x_{2}^2+16x_{1}^2+4x_{2}^2+1-64x_{1}^2x_{2}^2}\begin{bmatrix}
4x_{2}^2+1 & -8x_{1}x_{2} \\
-8x_{1}x_{2} & 16x_{1}^2+1
\end{bmatrix} \\
&= \frac{1}{16x_{1}^2+4x_{2}^2+1}\begin{bmatrix}
4x_{2}^2+1 & -8x_{1}x_{2} \\
-8x_{1}x_{2} & 16x_{1}^2+1
\end{bmatrix} \\
(\mathbf{J}_{f}^\top\mathbf{J}_{f})^{-1}\mathbf{J}_{f}^\top &= \frac{1}{16x_{1}^2+4x_{2}^2+1}\begin{bmatrix}
4x_{2}^2+1 & -8x_{1}x_{2} \\
-8x_{1}x_{2} & 16x_{1}^2+1
\end{bmatrix}\begin{bmatrix}
4x_{1} & 1 & 0 \\
2x_{2} & 0 & 1
\end{bmatrix} \\
&= \frac{1}{16x_{1}^2+4x_{2}^2+1} \begin{bmatrix}
16x_{2}^2x_{1}+4x_{1}-16x_{2}^2x_{1} & 4x_{2}^2+1 & -8x_{1}x_{2} \\
-32x_{1}^2x_{2}+32x_{1}^2x_{2}+2x_{2} & -8x_{1}x_{2} & 16x_{1}^2+1
\end{bmatrix} \\
&= \frac{1}{16x_{1}^2+4x_{2}^2+1}\begin{bmatrix}
4x_{1} & 4x_{2}^2+1 & -8x_{1}x_{2} \\
2x_{2} & -8x_{1}x_{2} & 16x_{1}^2+1
\end{bmatrix},
\end{align} $$
We see the determinant is strictly positive, as so it has an inverse.

### 3) Curve $c$, derive speed function and euclidean length
Consider the curve $c:[0,1]\rightarrow \mathbb{R}^2$ defined as
$$ c(t)=\begin{pmatrix}
2t+1 \\
-t^2
\end{pmatrix} $$
1) Derive an expression for the speed function $t\rightarrow ||\dot{c}_{t}||$
$$ \begin{align}
\dot{c}_{t} &=\begin{pmatrix}
2 \\
-2t
\end{pmatrix}, \\
||\dot{c}_{t}|| &= \sqrt{\begin{bmatrix}
2 & -2t
\end{bmatrix}^\top\begin{bmatrix}
2 \\
-2t
\end{bmatrix}} = \sqrt{ 4+4t^2+4 } = 2\sqrt{1 + t^2 } \\
\end{align} $$
2) Compute the euclidean length of the curve:
Since this is not the length of $f(c_t)$, we do not need the Jacobian of $f$, and can just do the integral of the above
$$ \begin{align}
\int_{0}^1||\dot{c}_{t}||\text{d}t &= \int_{0}^12\sqrt{ 1+t^2 }\text{d}t \\
&= 2\int_{0}^1\sqrt{ 1+t^2 }\text{d}t \\
&= 2\cdot\left( \left[ \frac{1}{2}(\sqrt{ 1+t^2 }t+\sinh^{-1}(t)) \right]_{0}^1 \right) \\
&= 2\cdot\left( \left[ \frac{1}{2}(\sqrt{ 1+1^2 }1+\sinh^{-1}(1) \right] - \left[ \frac{1}{2}(\sqrt{ 1+0^2 }0+\sinh^{-1}(0)) \right] \right) \\
&=2\cdot\left( \frac{1}{2}(\sqrt{ 2 }+\sinh^{-1}(1)) \right) \\
&=\sqrt{ 2 }+\sinh^{-1}(1)
\end{align} $$

### 4) 