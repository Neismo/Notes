	
# Flows for Continuous Random Variables

We consider a new way to model $p(\mathbf{x})$

>[!faq]+ Example (Change of Variable)
> Consider $z\in\mathbb{R}$ with $\pi(z) = \mathcal{N}(z|0, 1)$. Then consider a transformation of the variable with $x = 0.75\cdot z + 1$. *What is the distribution of $x$?*
> Instead of relying on information about Gaussian distributions, we can use the following:
> >[!faq]+ Change of Variable Method
> >$$ p(x) = \pi\left(z=f^{-1}(x)\right)\left|\frac{\partial f^{-1}(x)}{\partial x}\right| $$
> >We require that the function is *bijective*. We multiply with the *Jacobian* to preserve the density integrating to $1$!
> 
> In our example we have $f(z)=0.75\cdot z + 1$, and so $f^{-1}(x)=\frac{x-1}{0.75}$. Then the **change of volume** is:
> $$ \left|\frac{\partial f^{-1}(x)}{\partial x}\right| = \left|\frac{4}{3}\right| = \frac{4}{3} $$
> Putting *all* this together we get:
> $$ p(x) = \pi\left( z = \frac{x-1}{0.75} \right)\frac{4}{3} = \frac{1}{\sqrt{2\pi0.75^2}}\exp{\left(-\frac{(x-1)^2}{2\cdot0.75^2}\right)} = \mathcal{N}(x|1,0.75) $$
> This last step did insertion of the PDF of a Gaussian for $\pi(z)$ where $z = \frac{x-1}{0.75}$ and then simplifies (now shown)


The above example shows we can calculate a new distribution of a continuous random variable by applying a ==known bijective function== to a random variable with a ==known distribution==. The same holds for multiple variables $\mathbf{x,z}\in\mathbb{R}^D$:

$$ p(\mathbf{x}) = p(z=f^{-1}(\mathbf{x}))\left|\frac{\partial f^{-1}(\mathbf{x})}{\partial\mathbf{x}}\right|,\quad \left|\frac{\partial f^{-1}(\mathbf{x})}{\partial\mathbf{x}}\right|=|\det\mathbf{J}_{f^{-1}}(\mathbf{x})|,\tag{A} $$ ^df8c27

where $\mathbf{J_{f^{-1}}}$ is the ==Jacobian Matrix== defined as:
$$ \mathbf{J}_{f^{-1}} = \begin{bmatrix} \frac{\partial f^{-1}_1}{\partial x_1} & \cdots & \frac{\partial f^{-1}_1}{\partial x_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial f^{-1}_D}{\partial x_1} & \cdots & \frac{\partial f^{-1}_D}{\partial x_D} \end{bmatrix}. $$
^jacobidef

Further more, we have that the **inverse function theorem** to do:
$$ |\det\mathbf{J}_{f^{-1}}(\mathbf{x})| = |\det\mathbf{J}_{f(\mathbf{z})}|^{-1}, $$

which allows us re-write [(A)](#^df8c27):
$$ p(\mathbf{x}) = p\left( \mathbf{z} = f^{-1}(\mathbf{x})\right)|\det\mathbf{J}_{f(\mathbf{z})}|^{-1} $$
When the determinant of the Jacobian is $1$, we call it ==volume preserving==.

## Change of Variable for Deep Learning

Consider a **hierarchical model** with a sequence of invertible transformations $f_k:\mathbb{R^D}\rightarrow\mathbb{R^D}$. Starting with a *known distribution* $\pi(\mathbf{z}_0)=\mathcal{N}(\mathbf{z}_0|0,\mathbf{I})$, we apply the invertible functions in sequence to obtain a transformation:
$$ p(\mathbf{x}) = \pi(\mathbf{z}_0=f^{-1}(\mathbf{x}))\prod_{i=i}^K\left|\det\mathbf{J}_{f_i}(\mathbf{z}_{i-1})\right|^{-1} $$

In principle, we want to be able to get *any arbitrary distribution* and the ability to *revert to a simple one*! If we take the ==logarithm to above==. Due to the superscript of $-1$ on the Jacobians, we *subtract* the sum of them.
$$ \ln p(\mathbf{x}) = \underbrace{\ln \mathcal{N}(\mathbf{z}_0=f^{-1}(\mathbf{x})|0,\mathbf{I})}_{\text{Mean Square Error}} - \underbrace{\sum_{i=1}^K\ln|\det\mathbf{J}_{f_i}(\mathbf{z}_{i-1})|}_{\text{Reguralizer}}\tag{Log Likelihood} $$
^LearningObj

Why is it considered MSE? Because that term is *proportional* to $-||f(x) - 0||^2$ from the PDF of the Gaussian. This is exactly MSE!

1) How to model the invertible functions?
2) What is difficult here?

The first questions answer can be **Neural Networks** but not *any* NNs; they must be *invertible* and the *Jacobians* of them should be *easy to compute in logarithms*. These are called ==Normalizing Flows== or ==Flow-Based Models==! One such is **RealNVP** (Real-valued Non Volume Preserving) flows.

## Building Blocks for RealNVP

### Coupling Layers

The main component of RealNVP is ==the coupling layer==. Consider a split input $\mathbf{x} = [\mathbf{x}_a,\mathbf{x}_b]$; this may be done simply by splitting the vector $\mathbf{x}_{1:d}$ and $\mathbf{x}_{d+1:D}$ or similar techniques. Then *transform as follows*:
$$\begin{align}
\mathbf{y}_a &= \mathbf{x}_a \\
\mathbf{y}_b &= \exp(s(\mathbf{x_a}))\odot\mathbf{x}_b+t(\mathbf{x}_a),
\end{align}$$

Here $s(\cdot),t(\cdot)$ are *arbitrary neural networks* called *scaling* and *translation* respectively; their *inverses* are:
$$\begin{align}
\mathbf{x}_b &= (\mathbf{y}_b-t(\mathbf{y_a})) \odot\exp(-s(\mathbf{y}_a)) \\
\mathbf{x}_a &= \mathbf{y}_a
\end{align}$$

Importantly, the logarithm of the Jacobian becomes easy:
$$ \mathbf{J}=\begin{bmatrix} \mathbf{I}_{d\times d} & \mathbf{0}_{d\times(D-d)} \\ \frac{\partial\mathbf{y}_b}{\partial\mathbf{x}_a} & \text{diag}(\exp(s(\mathbf{x}_a))) \end{bmatrix},\quad \det(\mathbf{J}) = \prod_{j=1}^{D-d}\exp(s(\mathbf{x}_a))_j = \exp\left(\sum_{j=1}^{D-d}s(\mathbf{x}_a)_j\right) $$
^JacobianFormulation

The smart thing about the above Jacobian formulation, is despite the lower-left part $\frac{\partial y_b}{\partial x_a}$ is not trivial, because it is a *lower-triangle* we only need the *diagonal* for the determinant!

This transformation only processes around *d/D* parts of the input though! The other part are *permutation layers*:

### Permutation Layers

Here we consider ==permutation layers==, which are simple and effective too. Since permutations are *volume-preserving* by definition, their Jacobians are 1, so we apply them each time after a *coupling layer*. We could, for example, reverse order of variables!

![[CouplingPermuteFlow.png]]

## Dequantization

As introduced, we work with *real-valued continous* data; hoever, sometimes we work with *images* or other *discretely natured data*; for images we work in $x\in\{0,1,\cdots,255\}^D$. So to solve this, we can do ==uniform dequantization== with $\mathbf{x} = \mathbf{y}+\mathbf{u}$ where $u\in[-0.5,0.5]^D$ uniformly.

## Flows in Action

We learn with the *log-likelihood* function which we see [here from earlier](#^LearningObj). We used *coupling layers* combined with *permutation layers*. The *permutation layers* had Jacobian determinant of 1, so $\ln(1)=0$ simplifies it a bit:
$$ \ln p(\mathbf{x}) = \ln \mathcal{N}(\mathbf{z}_0=f^{-1}(\mathbf{x})|0,\mathbf{I}) -\sum_{i=1}^K\left(\sum_{j=1}^{D-d}s_k\left(\mathbf{x}_a^k\right)_j\right), $$

Where $s_k$ is the *scale network* in the $k$'th *coupling layer*, and $x_a^k$ denotes the *input*  to the $k$'th coupling layer. The log cancels out with the $\exp$ seen originally in the [Jacobian Fomulation](#^JacobianFormulation)

As alluded to earlier, the first term can be considered an MSE term; here is why when you plug in the PDF of the Gaussian:

$$ \ln \mathcal{N}(\mathbf{z}_0=f^{-1}(\mathbf{x})|0,\mathbf{I}) = \text{const} - \frac{1}{2}||f^{-1}(\mathbf{x})-0||^2,\quad\quad \frac{1}{2}||f^{-1}(\mathbf{x})-0||^2 = MSE(0,f^{-1}(\mathbf{x}))$$

When coding, for the Jacobian, we must remember to output *both the* output, but also the outcome of the *scaling layer* too, as it is used in the sum!


# Exercises

## 1) Linear Transformations are Volume-Preserving
_Consider a linear transformation $f : \mathbb{R}^D \rightarrow \mathbb{R}^D$ given by $f(\mathbf{x}) = −\mathbf{x} + \mathbf{a}$ for some $\mathbf{a} ∈ \mathbb{R}^D$. Show that this transformation is volume-preserving, c.f., section 4.1.1 of the textbook._
**Answer:** to show this is *volume-preserving*, we must calculate the *determinant* of the Jacobian for the *inverse function*:
$$ \mathbf{J}_{f^{-1}} = \begin{bmatrix} \frac{\partial f^{-1}_1}{\partial x_1} & \cdots & \frac{\partial f^{-1}_1}{\partial x_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial f^{-1}_D}{\partial x_1} & \cdots & \frac{\partial f^{-1}_D}{\partial x_D} \end{bmatrix}. $$
The inverse function is $f^{-1}(x)$ is actually itself! See $f^{-1}(f(x))=-(-x + a)+a = x-a+a=x$. It's Jacobian is also simple, as it is:
 $$ \mathbf{J}_{f^{-1}} = -\mathbf{I}_D, $$
 That is because we take the $(i,j)$'th entry to be $\frac{\partial f_i}{x_j}$ which is $1$ when $i=j$ and $0$ otherwise!
 The absolute value of the determinant of this is always equal to 1 of course!

## 2) Inverse of Sequence of Invertible Functions
Let $F(\mathbf{x}) = f_K\circ f_{K−1}\circ\cdots\circ f_2\circ f_1(\mathbf{x})$, where $f_k : \mathbb{R}^D \rightarrow \mathbb{R}^D$ for $k = 1,\cdots, K$ are invertible functions. Show that the inverse of F is given by $F^{−1}(\mathbf{z}) =f^{-1}_1\circ f^{−1}_2\circ\cdots\circ f^{−1}_{K−1}\circ f^{−1}_K(\mathbf{z})$.

First, say $K=1$, we have that $f^{1}_1\circ f_1(\mathbf{x})=\mathbf{x}$ and also $f_1\circ f_1^{\mathbf{z}}=\mathbf{z}$. To ease notation, let $I(x)=x$ be the identity function. Let us by induction show it holds for $K$ starting with the base case $K=2$.
The inverse is $F^{-1}(\mathbf{z}) = f_{1}^{-1}\circ f_2^{-1}$ as it holds that $f_{1}^{-1}\circ f_2^{-1} \circ f_2 \circ f_1(\mathbf{x})$ = $f_{1}^{-1}\circ I \circ f_1(\mathbf{x}) = I(\mathbf{x}) = x$.
Assume it holds for the base case, let us now show for $K>2$:
We write out $F = f_K\circ f_{K−1}\circ\cdots\circ f_2\circ f_1 = f_K\circ (f_{K−1}\circ\cdots\circ f_2\circ f_1) = f_K\circ G$, where $G=(f_{K−1}\circ\cdots\circ f_2\circ f_1)$. We showed in the base case that for any functions $A\circ B$ its inverse is $B^{-1}\circ A^{-1}$. So $F^{-1} = (f_K^\circ G)^{-1} = G^{-1}\circ f_K^{-1} = (f^{-1}_1\circ f^{−1}_2\circ\cdots\circ f^{−1}_{K−1})\circ f_K^{-1}$. We drop the parenthesis and we have:
$$F^{−1}(\mathbf{z}) =f^{-1}_1\circ f^{−1}_2\circ\cdots\circ f^{−1}_{K−1}\circ f^{−1}_K(\mathbf{z})$$
## 3) Jacobian of a Composed Function
Consider the function $h = g\circ f$ that is composed of the two functions $f:\mathbb{R}^D\rightarrow\mathbb{R}^D,\quad g:\mathbb{R}^D\rightarrow\mathbb{R}^D$. Show that $|\det\mathbf{J}_h| = |\det\mathbf{J}_f||\det\mathbf{J}_g|$ by the following steps:
1) Show that $\mathbf{J}_h=\mathbf{J}_g\mathbf{J}_f$
	1) Use the *chain rule* to write down the expression of $(i,j)$'th entry $\mathbf{J}_h$ i.e., $\frac{\partial h_i}{\partial x_j}$
	2) Use the definition of *matrix multiplication* to write down the $(i,j)$'th entry of $\mathbf{J}_g\mathbf{J}_f$.
2) Use that the determinant distributes over multiplication to arrive at the final results

First, the $(i,j)$'th entry in $\mathbf{J}_h$ can be expressed as $\nabla g_i(f(\mathbf{x}))\cdot\frac{\partial f}{x_j}$; If we combine this the matrix looks like:
$$ \mathbf{J}_h = \begin{bmatrix} 
\nabla g_1 \cdot \frac{\partial f}{\partial x_1} & \nabla g_1 \cdot \frac{\partial f}{\partial x_2} & \cdots & \nabla g_1 \cdot \frac{\partial f}{\partial x_D} \\ \nabla g_2 \cdot \frac{\partial f}{\partial x_1} & \nabla g_2 \cdot \frac{\partial f}{\partial x_2} & \cdots & \nabla g_2 \cdot \frac{\partial f}{\partial x_D} \\ \vdots & \vdots & \ddots & \vdots \\ \nabla g_D \cdot \frac{\partial f}{\partial x_1} & \nabla g_D \cdot \frac{\partial f}{\partial x_2} & \cdots & \nabla g_D \cdot \frac{\partial f}{\partial x_D} \end{bmatrix} $$
We can reason that $\mathbf{J_g}$ has *rows* that are the *gradient* of it's fucntions so each row is $\nabla g_i$, whereas each column in $\mathbf{J}_f$ has the *partial derivative vector* $\partial f/\partial x_i$. We know that the matrix product $AB$ is a dot product between rows and columns! (i'th row with j'th column).

To get a better understanding of ___why___ we can do this, consider that we can write the Jacobian arbitrarily as seen in the [definition](#^jacobidef). If we look at _a column_, then we see it is the different $f_i$ functions partial derivative at ___the same___ $x_k$. Considering rows, each _row_ is one $f_i$ derivation at ___all the different___ $x_k$'s. Respectively, we can summarize the *columns* as $\frac{\partial f}{\partial x_k}$ and the rows as $\nabla f_i = \left( \frac{\partial f_i}{\partial x_1},\cdots,\frac{\partial f_i}{\partial x_D} \right)$.

This allows us to decompose the original $\mathbf{J}_h$ into the following,
$$\mathbf{J}_h = \mathbf{J}_g\mathbf{J}_f = \begin{bmatrix} \nabla g_1 \\ \nabla g_2 \\ \vdots \\ \nabla g_D \end{bmatrix}\times\left[ \frac{\partial f}{\partial x_1}\; \frac{\partial f}{\partial x_2}\;\cdots\; \frac{\partial f}{\partial x_D} \right]. $$

Furthermore, we use that the $\det(AB) = \det(A)\det(B)$ to distribute it over the two. Then it follows that:
$$|\det(\mathbf{AB})|=|\det(\mathbf{A})\det(\mathbf{B})|=|\det(\mathbf{A})||\det(\mathbf{B})|$$
So if it easier to calculate the Jacobian's for each of the composing functions (and their determinants), then it is possible to do that instead of the Jacobian from the composed function.