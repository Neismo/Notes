
# Examples of Learned Matrices

We have considered models with manifolds through  a mapping $f:\mathbb{R}^d\to \mathbb{R}^D$, but it can be modeled directly. Consider data $\mathbf{y}_{1:N} = \{\mathbf{y}_{1},\cdots, \mathbf{y}_{N}\}$, where $\mathbf{y}_{i}\in \mathbb{R}^D$, and use developed tools directly. A simple approach is
$$ \mathbf{G}_{\mathbf{y}} = g(\mathbf{y})\mathbf{I}, $$
where $g:\mathbb{R}^D\to \mathbb{R}_{+}$ is a learned function that controls the *Metric*; recall the pull-back Metric is
$$  \mathbf{G}_{\mathbf{x}}=\mathbf{J}_{\mathbf{x}}^\top\mathbf{J}_{\mathbf{x}}.  $$
If $g$ is smooth, then $\mathbf{G_{y}}$ is also smooth. Since $g$ is positive, then $\mathbf{G_{y}}$ is positive definite, and it is a Riemannian metric (which requires to be smooth and positive definite).

As an example, consider modeling $g$ as the inverse of the data density
$$g(\mathbf{y}) = \frac{1}{p(\mathbf{y})},$$
where $p(\mathbf{y})$ is the density of the data. In this case, we will have low values in areas near the data, but high outside. This leads distances to be smaller in the data region, but further in sparse regions. This is due to how we defined the lengths to be among the infinitesimal manifold:
$$ \langle \delta_{1},\delta_{2} \rangle_{\mathbf{x}} = \delta_{1}^\top\mathbf{G}_{\mathbf{x}}\delta_{2}. \tag{As defined previously}$$

![[ExampleG.png]]

As can be seen above, the distances with said $g$ produces shortest paths *alongside* the density, and not the direct euclidean.

## Autoencoders

An *autoencoder* is prototypical of manifolds. Has *encoder* that approximates the mapping to a manifold: $g:\mathbb{R}^D\to \mathbb{R}^d$ and a *decoder* which describes the manifold $f:\mathbb{R}^d\to \mathbb{R}^D$. The manifold is then approximated with the loss
$$ \sum_{n=1}^N||\mathbf{x}_{n} - f(g(\mathbf{x}_{n}))||_{2}^2. $$
Here we see the same previous example, which also illustrated how on the manifold the ==geodesic distance== is compared to the euclidean; it avoid steep hills.
![[ExampleGAutoEncoder.png]]
## Classification Metrics
Riemannian metrics can also shed light on classification; say we have $K$ classes, and we have a classifier $f:\mathbb{R}^D\to C$, where $C$ is a vector space of dimension $K$; they sum to 1.

The metric is still then
$$\mathbf{G_{x}} = \mathbf{J}_{\mathbf{x}}^\top\mathbf{J_{x}}$$
but for this to be positive definite, the rank of $\mathbf{J}_{x}$ must have rank equal to the data-dimension. The rank of the Jacobi is $\text{min}(K-1,D)$, so we must have more classes than data dimensions. This is possible in 2D as we then have 3 classes, but not in 3D; see example here
![[ClassificatioNG.png]]
which highlights the values of the determinant of the metric; it is much different around the decision boundary. It breaks down in higher dimensions, which is a common situation to be in.

In that case a semi-positive definite metric (some eigenvalues zero) still allows the use of some of the tools we have so far. Then it is ==pseudo-Riemannian==. In this case, some distinct points may have a distance of $0$; which is counter-intuitive.

# Geodesics
Recall, the shortest path on a manifold is the ==geodesic==.
### Uniqueness
Assume the earth is a sphere, which would be the manifold. If we have two points on it, the arc between them on the sphere is the length; is it unique? No, what if we have two *antipodal* points - one on north pole, one on south pole. Then there are *infinitely* many arcs between them that are equally short.

$$\begin{align}
c^* &= \arg\min_{c}\text{Length}_{\mathcal{M}}(c) \tag{1} \\
& \text{subject to }c(0)=\mathbf{x}_{n} \text{ and } c(1)=\mathbf{x_{m}}
\end{align}$$

All curves are geodesics, and we may have several local optima.

## Constant Speed Parametrizations

Geodesics are not unique; and if we look on the equation above (1), we seek a curve of minimal length. It can have parameters that change the curve, and if for any set of them evaluate the curve length for those parameters and visualize it. 
It may look like this: if I walk slow and then fast, I get the ==same curve ==if I walk fast, then slow.
![[GeodesicsParameters.png]]
In reality, we have infinitely many parameters, which is against the drawing here, but it is obvious that the landscape is not simple.

There may be two good ways to connect points; while we can't do anything against that, the *reparametrization issue* is something we can do something about, which makes it **go away**. When two parameter choices give rise to the same curve, ==choose among those arbitrarily and stick to it==; we make the *simple* choice; move at a ==*constant speed*==, independent of the time step $t$
$$ \begin{align}
\text{Length}_{\mathcal{M}}(c)&= \int_{0}^1 ||\dot{c}_{t}||_{2}^2\text{d}t \\
&= \int_{0}^1 v\text{d}t \\
&= v\int_{0}^1 \text{d}t=1,
\end{align} $$
where $v=||\dot{c}_{t}||$ is the *constant velocity* independent of $t$. This gives us *unique curves*, but ==length computations are also very simple now==.

We will see how to construct those curves now

### Curve Energy
Instead of finding shortest path by minimizing length, we will formulate a similar minimization problem whose solution are also shortest paths.

Recall the ==Cauchy-Schwarz== inequality
$$|\langle u,v\rangle| \leq ||u||_{2}\cdot||v||_{2},$$
which is only equal when $u,v$ are parallel. Also recall $||u|| = \sqrt{\langle u,u\rangle }$ We will use this to bound the length of a curve as, and the usual choice is
$$ \langle u,v\rangle = \int_{0}^1u_{t}v_{t}\text{d}t $$
for curves $u,v:[0,1]\to \mathbb{R}$. To bound the length of a curve $c$, we introduce the curve $u:[0,1]\to\mathbb{R}$ that measures the speed of $c$
$$ u_{t} = ||\dot{c}_{t}||_{2} $$
For seemingly silly reasons, we also introduce the constant curve $v:[0,1] \to \mathbb{R}$
$$ v_{t} = 1, $$

So now the length of the curve can be written as
$$ \text{Length}_{\mathcal{M}}(c) = \int_{0}^1u_{t}v_{t}\text{d}t = \int_{0}^1||\dot{c}_{t}||_{2}\text{d}t = \langle u,v\rangle. $$
By the **Cauchy-Schwarz** inequality (curve is always positive due to norm, so no absolute value bars)
$$ \begin{align}
\text{Length}_{\mathcal{M}}(c) &\leq ||u||\cdot||v|| \\
&= \sqrt{ \int_{0}^1 ||\dot{c}_{t}||_{2}^2 \text{d}t}\sqrt{ \int_{0}^1 1^2 \text{d}t } \\
&= \sqrt{ \int_{0}^1 ||\dot{c}_{t}||_{2}^2 \text{d}t } \tag{2},
\end{align} $$

The last integral is important as is given the name of *energy* of the curve *c* and denote it
$$ \mathcal{E}(c) = \int_{0}^1||\dot{c}_{t}||_{2}^2\text{d}t, \tag{Energy} $$
Squaring both sides in (2) above yields
$$ \text{Length}^2_{\mathcal{M}}(c) \leq \mathcal{E}(c), $$
which is from the *cauchy-schwarz* inequality. It is equal only when $u,v$ are parallel, and proportional to one-another:
$$ u_{t}=\alpha v_{t} \Leftrightarrow ||\dot{c}_{t}|| = \alpha, $$
which means ==equality only happens when the speed is constant==. So ==minimize curve energy== is minimizing the path with constant speed - in the optimization landscape we have *local optimums*
![[TakeHomeWeek6.png]]
![[EnergyMinimizationOptim.png]]
## Governing Differential Equations
We also require ODEs to capture behaviour of Geodesics. The ==Euler-Lagrange== theorem states that integrals on the form

$$ \int_{0}^1 E(t,c_{t},\dot{c}_{t})\text{d}t, $$
where $E:\mathbb{R}\times \mathbb{R}^d\times \mathbb{R}^d\to \mathbb{R}$ are minimized with respect to a curve *c* when
$$ \frac{\partial E_{t}}{\partial c_{t}} = \frac{\text{d}}{\text{d}t}\frac{\partial E_{t}}{\partial \dot{c}_{t}}, $$
In our particular energy curve we have
$$ E_{t} = E(t,c_{t},\dot{c}_{t}) = \dot{c}_{t}\mathbf{G}_{c_{t}}\dot{c}_{t}, $$
The results are then
$$ \begin{align}
[\ddot{c}_{t}]_{m} &= -\sum_{j=1}^d\sum_{k=1}^d \Gamma^m_{jk}(c_{t})[\dot{c}_{t}]_{j}[\dot{c}_{t}]_{k} \\
\Gamma^m_{jk}(c_{t}) &=\frac{1}{2}\sum_{i=1}^d[\mathbf{G}_{c_{t}}^{-1}]_{mi}\left( 2 \frac{\text{d}[\mathbf{G}_{c_{t}}]_{ij}}{\text{d}[c_{t}]_{k}} - \frac{\text{d}[\mathbf{G}_{c_{t}}]_{jk}}{\text{d}[c_{t}]_{i}} \right)
\end{align} $$

Derivation in the book for this (lots of re-writing). A solution with vectors instead of indices is also possible

$$ \ddot{c}_{t} = -\frac{1}{2}\mathbf{G}_{c_{t}}^{-1}\{ 2(\mathbf{I} \otimes \dot{c}_{t})\partial_{c_{t}}\text{vec}(\mathbf{G}_{c_{t}})\dot{c}_{t}-\partial_{c_{t}} \text{vec}(\mathbf{G}_{c_{t}})^\top (\dot{c}_{t}\otimes \dot{c}_{t}) \}, $$
where $\text{vec}(\circ)$ stacks columns of a matrix and $\otimes$ is Kronecker product.

### Shooting Geodesics
This allows us to see where we can end up with an initial position $c_{n}$ and initial velocity $\dot{c}_{t}$. Above tells us we can solve it.

# Computing Geodesics Numerically
In practice, we want algorithms; none are very good however. In practice, we minimize the *energy* of the curve. Summarized as
>[!note] Computing Connecting Geodesics
>1) Parametrize a curve that connects $c_0$ and $c_1$.
>2) Numerically approximate the energy of the parametric curve, and finally
>3) Minimize the energy using gradient-based optimization.

## Step 1; piecewise linear curves OR polynomials

Partition interval into $N+1$ equidistant points; endpoints are fixed, middle points are free parameters.
$$ t_{i} = \frac{i}{N},\quad\text{for } i \text{ in}= 0,\cdots,N $$
so for $N=3$ we get $\left[0, \frac{1}{3}, \frac{2}{3}, 1 \right]$. For each $t_i$ we keep an associated position along the curve $\tilde{c}_{i}\in \mathbb{R}^d.$ The endpoints are known which are $\tilde{c}_{0}=c_{0},\tilde{c}_{N}=c_{1}$ and are fixed, while the remaining positions are *free parameters*. It is with these they are optimized.

![[FreeParam.png]]

Smooth manifolds may not befit a linear piecewise function. A polynomial function could also work, and any parametric curve work (splines, bezier); the curve is split into linear and non-linear parts:
$$ c(t) = (1-t)c_{0} + tc_{1} + \tilde{c}(t), $$
where $\tilde{c}(t)$ is to have constructed endpoints at zero so $\tilde{c}(0)=\tilde{c}(1)=0$. It can be construced as a polynomial as well
$$ \tilde{c}_{i}(t)=\sum_{k=1}^Kw_{ik}t^k_{0}w_{i_{0}}, $$
and this is 0 when $w_{i_{0}}=0,w_{iK} = -\sum_{k=1}^{K-1}w_{ik}$

## Step 2; Energy Quadrature & Monte Carlo

Riemann Sum of integral of the energy
 $$ \mathcal{E}(c) = \int_{0}^1||\dot{c}_{t}||_{\mathbf{G}_{c_{t}}}^2\text{d}t \approx \tilde{\mathcal{E}}(c) = \sum_{s=0}^S||\dot{c}_{s}||^2_{\mathbf{G}_{c_{s}}}\Delta s, $$
 where $c_{s} = c\left( \frac{s}{S} \right)$ are place on a regular grid over the interval $[0,1]$ and $\Delta s=\frac{1}{s+1}$ .
This is easy with access to decoder we can approximate the above with:
$$ \tilde{\mathcal{E}}(c) = \sum_{s=1}^S||f(c_{s}) - f(c_{s-1})||_{2}^2 $$
# Arithmetic in latent space
Euclidean doesn't work, so what can we do. Where $\mathcal{M}$ is points, and $T$ are vectors

$$ \begin{align}
-&:\mathcal{M}\times\mathcal{M}\to T \\
+&:\mathcal{M}\times T\to\mathcal{M} \\
\Delta&= \mathbf{x}_{n}-\mathbf{x}_{m} \\
\text{dist}(\mathbf{x}_{n},\mathbf{x}_{m}) &= || \mathbf{x}_{n} - \mathbf{x}_{m} ||_{2} = ||\Delta||_{2}
\end{align} $$

**Subtraction**: we know shortest path is defined by ODE as starting point and starting velocity; subtraction in euclidean arithmetic gives us shortest path ($\Delta$); so similarly we have
$$ \text{Log}_{\mathbf{x}_{m}}(\mathbf{x}_{n})=\Delta=\partial_{t}c_{mn}(0), $$
(this belongs to the *tangent space*); this is called the ==logarithm map== (Not actual log function).

**Addition**: is *inverse* subtraction. Thus we define it as
$$\text{Exp}_{\mathbf{x}_{m}}(\Delta) = \text{Exp}(\partial_{t}c_{mn}(0)) = \mathbf{x}_{n}$$
![[AddtionSubtractionRiemannian.png]]
# Theoretical Exercises

## 1) Quadratic metric
Consider a two-dimensional abstract manifold with the metric
$$ \mathbf{G}_{\mathbf{x}} = (1 + ||\mathbf{x}||_{2}^2)\mathbf{I}, \mathbf{x}\in \mathbb{R}^2 $$
and consider the points
$$ \mathbf{x}_{1} = \begin{pmatrix}
1 \\
1
\end{pmatrix},\quad \mathbf{x}_{2} = \begin{pmatrix}
2 \\
3
\end{pmatrix},\quad \mathbf{x}_{3}=\begin{pmatrix}
0 \\
3
\end{pmatrix} $$

1) Compute the local norms of the tangent vector $v_1=(1\quad 0)^\top$ assuming the point of tangency is $\mathbf{x}_1$, $\mathbf{x}_2$ and $\mathbf{x}_3$, respectively
$$ ||\mathbf{v}||_{\mathbf{x}} =\sqrt{ \langle\mathbf{v},\mathbf{v}\rangle_{\mathbf{x}} } = ||\mathbf{Jv}||_{2} = \sqrt{ \mathbf{v}^\top\mathbf{J}_{x}^\top\mathbf{J}_{x}\mathbf{v} } = \sqrt{ \mathbf{v}^\top\mathbf{G}_{\mathbf{x}}\mathbf{v}  }$$
we insert our values for the three points and evaluate
$$ \begin{align}
||\mathbf{v}_{1}||_{\mathbf{x}_{1}} &= \sqrt{ [1\ 0]\mathbf{G}_{\mathbf{x}_{1}}\begin{bmatrix}1 \\ 0\end{bmatrix} } = \sqrt{ [1\ 0]\begin{bmatrix}1+2 & 0 \\0 & 1+ 2\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} } = \sqrt{ 3 } \\
||\mathbf{v}_{1}||_{\mathbf{x}_{2}} &= \sqrt{ [1\ 0]\mathbf{G}_{\mathbf{x}_{2}}\begin{bmatrix}1 \\ 0\end{bmatrix} } = \sqrt{ [1\ 0]\begin{bmatrix}1+13 & 0 \\0 & 1+ 13\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} } = \sqrt{ 14 } \\
||\mathbf{v}_{1}||_{\mathbf{x}_{3}} &= \sqrt{ [1\ 0]\mathbf{G}_{\mathbf{x}_{1}}\begin{bmatrix}1 \\ 0\end{bmatrix} } = \sqrt{ [1\ 0]\begin{bmatrix}1+9 & 0 \\0 & 1+ 9\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} } = \sqrt{ 10 }
\end{align} $$
2) Compute the local angles between $v_1$ and $v_{2} = (0\ 1)^\top$ in the same three points of tangency.

We can find the angle like so
$$ \begin{align}
\theta &= \cos^{-1}\left( \frac{\langle\mathbf{v}_{1},\mathbf{v}_{2}\rangle_{\mathbf{x}}}{||\mathbf{v}_{1}||_{\mathbf{x}}\cdot||\mathbf{v}_{2}||_{\mathbf{x}}} \right) \\
&=\cos^{-1}\left( \frac{\mathbf{v}_{1}^\top\mathbf{J}_{\mathbf{x}}^\top\mathbf{J}_{\mathbf{x}}\mathbf{v}_{2}}{||\mathbf{J}_{\mathbf{x}}\mathbf{v}_{1}||\cdot||\mathbf{J}_{\mathbf{x}}\mathbf{v}_{2}||} \right)
\end{align} $$
Let us first evaluate at $\mathbf{x}_{1}$
$$ \begin{align}
\theta_{1} &= \cos^{-1}\left( \frac{\mathbf{v}_{1}^\top\mathbf{J}_{\mathbf{x}}^\top\mathbf{J}_{\mathbf{x}}\mathbf{v}_{2}}{||\mathbf{J}_{\mathbf{x}_{1}}\mathbf{v}_{1}||\cdot||\mathbf{J}_{\mathbf{x}_{1}}\mathbf{v}_{2}||} \right) \\
&= \cos^{-1}\left( \frac{[1\ 0]\begin{bmatrix}
3 & 0 \\
0 & 3
\end{bmatrix}\begin{bmatrix}
0 \\
1
\end{bmatrix}}{||\mathbf{J}_{\mathbf{x}_{1}}\mathbf{v}_{1}||\cdot||\mathbf{J}_{\mathbf{x}_{1}}\mathbf{v}_{2}||} \right)  \\
&= \cos^{-1}\left( \frac{0}{||\mathbf{J}_{\mathbf{x}_{1}}\mathbf{v}_{1}||\cdot||\mathbf{J}_{\mathbf{x}_{1}}\mathbf{v}_{2}||} \right) = \cos^{-1}(0) = \frac{\pi}{2}
\end{align}$$
Irregardless of the values we evaluate the metric at, we are going to have a 0 in the numerator and get a angle of $\frac{\pi}{2}$.

## 2) Euclidean Metric
Consider now $\mathbf{G}_{\mathbf{x}} = \mathbf{I} \in \mathbb{R}^d$, so independent of where we evaluate it.

1) Derive the coefficients of the geodesics ODE of this metric

All the coefficients in this case will be $0$ as the derivative of the metric (which is constant) is $0$, and so all evaluate to $0$.

2) Derive the geodesic ODE
As all coefficients are 0, then the ODE is 0 as well

3) What is the geodesic that connects points $\mathbf{x}_{1}$ and $\mathbf{x}_2$?
Integrating $0$ twice with regards to $t$ yields the straight line $v_{1}t+x_{0}$

## 3) Quadratic metric
Again consider $$\mathbf{G}_{\mathbf{x}} = (1 + ||\mathbf{x}||_{2}^2)\mathbf{I}\in \mathbb{R}^d, $$
1) Derive the coefficients of the geodesics ODE of this metric

As $$ \mathbf{G}_\mathbf{x}^{-1} = \frac{1}{1+||\mathbf{x}||_{2}^2}\mathbf{I} $$
We get that $$ \frac{\partial[\mathbf{G}_{\mathbf{x}}]_{ij}}{\partial [c_{t}]_{k}} = 2[c_{t}]_{k}\mathbf{I}_{ij}, $$
which when inserted gives us $c_{t} = \mathbf{x}$
$$ \begin{align}
\Gamma^m_{jk}(c_{t}) &=\frac{1}{2}\sum_{i=1}^d\frac{1}{1+||\mathbf{x}||_{2}^2}\mathbf{I}_{mi}\left( 2 \frac{\text{d}[\mathbf{G}_{c_{t}}]_{ij}}{\text{d}[c_{t}]_{k}} - \frac{\text{d}[\mathbf{G}_{c_{t}}]_{jk}}{\text{d}[c_{t}]_{i}} \right) \\
&= \frac{1}{2}\sum_{i=1}^d\frac{1}{1+||\mathbf{x}||_{2}^2}\mathbf{I}_{mi}\left( 2\cdot 2[c_{t}]_{k}\mathbf{I}_{ij} - 2[c_{t}]_{i}\mathbf{I}_{jk} \right) \\
&= \frac{1}{2}\sum_{i=1}^d\frac{\delta_{mi}}{1+||\mathbf{x}||_{2}^2}\left( 2\cdot 2[c_{t}]_{k}\delta_{ij} - 2[c_{t}]_{i}\delta_{jk} \right) \\ \\
&= \frac{1}{2}\frac{\left( 4[c_{t}]_{k}\delta_{mj} - 2[c_{t}]_{m}\delta_{jk} \right)}{1+||\mathbf{x}||_{2}^2} \\
&= \frac{\left( 4[c_{t}]_{k}\delta_{mj} - 2[c_{t}]_{m}\delta_{jk} \right)}{2\cdot(1+||\mathbf{x}||_{2}^2)} \\
&= \frac{ 2[c_{t}]_{k}\delta_{mj} - [c_{t}]_{m}\delta_{jk}}{1+||\mathbf{x}||_{2}^2}
\end{align} $$
2) Derive the Geodesic ODE
$$ [\ddot{c}_{t}]_{m} = -\sum_{j=1}^d\sum_{k=1}^d \Gamma^m_{jk}(c_{t})[\dot{c}_{t}]_{j}[\dot{c}_{t}]_{k} = -\sum_{j=1}^d\sum_{k=1}^d \frac{ 2[c_{t}]_{k}\delta_{mj} - [c_{t}]_{m}\delta_{jk}}{1+||c_{t}||_{2}^2}[\dot{c}_{t}]_{j}[\dot{c}_{t}]_{k} $$
3) Consider a geodesic $c$ starting at $c_0 = \mathbf{0}$ and initial velocity $\dot{c}_{0} = \mathbf{v}$. What is the acceleration $\ddot{c}_{0}$?