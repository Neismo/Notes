
# Moments of the multidimensional variables
Assume given a ==**random** or **stochastic** **matrix**==:
$$ \mathbf{X}=\begin{bmatrix}
X_{11} & \cdots & X_{1k} \\
\vdots & \ddots & \vdots \\
X_{n 1} & \cdots & X_{nk}
\end{bmatrix} $$

### Mean / expectation

then the **mean value** or **expectation** or **expected value** of **X** is:
$$ \text{E}(\mathbf{X}) = \begin{bmatrix}
\text{E}(X_{11}) & \cdots & \text{E}(X_{1k}) \\
\vdots & \ddots & \vdots \\
\text{E}(X_{n1}) & \cdots & \text{E}(X_{nk}) 
\end{bmatrix} = \begin{bmatrix}
\mu_{11} & \cdots & \mu_{1k} \\
\vdots & \ddots & \vdots \\
\mu_{n 1} & \cdots & \mu_{nk} 
\end{bmatrix} $$

>[!note]+ Theorem 1.1
>Let $\mathbf{A}$ be a $n \times n$ matrix of constants. Then
>$$ \text{E}(\mathbf{A}+\mathbf{X}) = \mathbf{A} + \text{E}(\mathbf{X}) $$

>[!note]+ Theorem 1.2
>Let $\mathbf{A}$ and $\mathbf{B}$ be constant matrices so that $\mathbf{AX}$ and $\mathbf{XB}$ exists. Then
>$$ \text{E}(\mathbf{AX}) = \mathbf{A}\text{E}(\mathbf{X}) $$
>$$ \text{E}(\mathbf{XB}) = \text{E}(\mathbf{X})\mathbf{B} $$

>[!note]+ Theorem 1.3
>Let $\mathbf{X}$ and $\mathbf{Y}$ be random matrices of the same dimension. Then
>$$ \text{E}(\mathbf{X} + \mathbf{Y}) = \text{E}(\mathbf{X}) + \text{E}(\mathbf{Y}) $$

### The variance-covariance matrix (==Dispersion matrix==)

The generalization of the variance of a random variable $\mathbf{X}=(X_{1},\cdots,X_{n})$ is the **==dispersion matrix==**

$$ \text{D}(\mathbf{X}) = \Sigma = \text{E}\{ (\mathbf{X}-\mathbf{\mu})(\mathbf{X}-\mathbf{\mu})^\top \}, $$
where $\mu=\text{E}(\mathbf{X})$ the expectation of the *random* matrix. We often use the notation:

$$ \Sigma = \begin{bmatrix}
\sigma_{1}^2 & \sigma_{12} & \cdots & \sigma_{1n} \\
\sigma_{21} & \sigma_{2}^2 & \cdots & \sigma_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{n 1} & \sigma_{n2} & \cdots & \sigma_{n}^2 
\end{bmatrix} $$
where we can also use $\sigma_{1}^2=\sigma_{11}$.

>[!note]+ Theorem 1.5
>The dispersion matrix $\Sigma$ for a multidimensional random variable is positive semidefinite. This is a *neccesary* and *sufficient* condition. That is, for all compatible vectors $y$
>$$ \mathbf{y}^\top\Sigma \mathbf{y} \geq 0 $$

>[!note] Theorem 1.6
>Let $\mathbf{X}$ and $\mathbf{Y}$ be independent. Then
>$$ \text{D}(\mathbf{X}+\mathbf{Y}) = \text{D}(\mathbf{X}) + \text{D}(\mathbf{Y}) $$
>Let $b$ be a constant. Then we have
>$$ \text{D}(b+\mathbf{X}) = \text{D}(\mathbf{X}) $$
>If $\mathbf{A}$ is a constant matrix, so that $\mathbf{AX}$ exists, then
>$$ \text{D}(\mathbf{AX}) = \mathbf{A}\text{D}(\mathbf{X})\mathbf{A}^\top $$

### Correlation
$$ \mathbf{V} = \text{diag}\left( \frac{1}{\sigma_{1}},\cdots,\frac{1}{\sigma_{n}} \right), $$
so a diagonal matrix with the standard deviations. If we *scale* $\mathbf{X}$ by $\mathbf{V}$ we get:

$$ \text{D}(\mathbf{VX}) = \mathbf{V\Sigma \mathbf{V^\top}} = \begin{bmatrix}
1 & \frac{\sigma_{12}}{\sigma_{1}\sigma_{2}} & \cdots & \frac{\sigma_{1n}}{\sigma_{1}\sigma_{n}}\\
\frac{\sigma_{12}}{\sigma_{1}\sigma_{2}} & 1 & \cdots & \frac{\sigma_{2n}}{\sigma_{2}\sigma_{n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\sigma_{1n}}{\sigma_{1}\sigma_{n}} & \frac{\sigma_{2n}}{\sigma_{2}\sigma_{n}} & \cdots & 1
\end{bmatrix} $$
This is the ==*correlation matrix*==, and we write:
$$ \text{R}(\mathbf{X}) = \begin{bmatrix}
1 & \cdots & \rho_{1n} \\
\vdots & & \vdots \\
\rho_{n 1} & \cdots & 1
\end{bmatrix}$$
where
$$ \rho_{ij} = \text{Corr}(X_{i}, X_{j}) = \frac{\text{Cov}(X_{i},X_{j})}{\sqrt{ \text{V}(X_{i}) \text{V}(X_{j}) }} $$
>[!faq] Remark 1.7
>The correlation matrix is *also* positive semidefinite.

### Covariance
Let there be given two random variables
$$ \mathbf{X} = \begin{bmatrix}
X_{1} \\ \vdots \\ X_{p}
\end{bmatrix} \qquad \mathbf{Y} = \begin{bmatrix}
Y_{1} \\ \vdots \\ Y_{q}
\end{bmatrix},$$
with mean values $\mathbf{\mu}$ and $\mathbf{v}$ respectively. The *covariance* between $\mathbf{X}$ and $\mathbf{Y}$ is then
$$ \text{C}(\mathbf{X,Y}) = \text{E}[(\mathbf{X-\mu})(\mathbf{Y-v})^\top] = \begin{bmatrix}
\text{Cov}(X_{1},Y_{1}) & \cdots & \text{Cov}(X_{1},Y_{q}) \\
\vdots & & \vdots \\
\text{Cov}(X_{p},Y_{1}) & \cdots & \text{Cov}(X_{p},Y_{q})
\end{bmatrix} $$
Then
$$ \text{C}(\mathbf{X,X}) = \text{D}(\mathbf{X}), $$
and 
$$ \text{C}(\mathbf{X,Y}) = [\text{Cov}(\mathbf{Y,X})]^\top $$
![[Theorem18.png]]

If $\text{C}(\mathbf{X,Y})=\mathbf{0}$, then we say the two are *uncorrelated*.

## The Multivariate Normal Distribution

![[theorem1.27.png]]

### Test for correlation
![[CorrelationTest.png]]
And CIs
![[CorrelationCI.png]]