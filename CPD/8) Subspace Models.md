# PCA
Rotate such that *projected data* has *maximum variance* along axis', in diminishing manner.
1) Center the data
$$ X = Z - \mu_{z} $$
2) Rotate coordinate system (not data); project onto the first axis of maximal variance:
$$ S = XL $$
3) New coordinates; so we can get 2D for example with:
$$ s_{i} = x_{i}L,\quad\quad s_{i} = (s_{i_{1}},s_{i_{2}}) $$

## Derivation
1) The transformation
$$ S = XL,\quad \text{where } L^\top L=I $$
2) Maximize Variance of projected data => maximize variance of each PC
$$ \text{cov}(S)=\frac{1}{n}S^\top S = \frac{1}{n}L^\top X^\top XL = \frac{1}{n}L^\top \Sigma L,\quad \Sigma=\text{Cov}(X). $$
3) The first PC objective
$$ \arg\max_{l} l^\top\Sigma l,\text{ such that } l^\top l=1 $$
4) Lagrangian formulation (to prevent $l$ from exploding)
$$ L_{p} = l^\top \Sigma l - \lambda(l^\top l-1) $$
5) Vector Calculus; take partial derivatives to $l_{p}$ and set it to zero:
$$ \frac{\partial L_{p}}{\partial l}=2\Sigma l-2\lambda l=0. $$
6) Eigenvalue decomposition; rearranging above we get
$$ \Sigma l=\lambda l, $$
**Mathematical conclusion**: covariance is *maximized* when *l* is the eigenvector of $\Sigma$. The maximal variance is exactly its corresponding eigenvalue $\lambda$! Because $\Sigma$ is symmetric, the eigenvectors are automatically orthogonal.

**Loading & Scores**
$$S = XL$$
**Loadings (L)**: The principal axis, a rotation matrix; columns are orthogonal and unit length
**Scores (S)**: coordinates of the data points along new axis

**Number of components**: choose those with eigenvalues greater than 1; alternatively, compute PCA eigenvalues for random data in same dimensionality, and choose those who are greater than in that plot. (*spree plot*)

SVD > Eigenvalue Decomposition (EVD) when *p>n* due to the fact that in EVD requires computing $X^\top X$.

![[Scree Plot.png]]

## Benefits & Properties
1) Uniqueness & Independence
   The *only* linear transformation with mathematically independent loading vectors and uncorrelated scores
2) Optimality in Compression
   Minimizes reconstruction error with PCA
3) Computational Efficiency
   Easy and quick to calculate with SVD
   Solves multicollinearity natively
   Handles $p>n$ seamlessly

**Drawback**: unsupervised nature leads to lack of interpretability in some cases.

# Sparse PCA
Want LASSO on PCA. Threshold such that if $l_{j}<\delta$, then set them to 0. This destroys the guarantees of PCA, which means we may have correlation, and the vectors may not be perpendicular.

# Principal Component Regression (PCR)
Linear Regression on PCA scores. PCR performs similar to ridge regression. Take scores and some $M$ such that we have scores $s_{1},\dots,s_{M}$ with $M\leq p$. Then standard regression
$$ y=\beta_{0}+[s_{1},\dots s_{M}]\beta+\epsilon. $$

Equivalent to OLS when $M=p$.

# Partial Least Squares (PLS)
Supervised method with latent variable structure. Seeks direction which have *high variance* and have *high correlation with target*!

Solve
$$ \begin{align}
&\max_{\alpha}\text{Corr}^2(y,X\alpha)\text{Var}(X\alpha) \\ \\
\text{ subject to }&||\alpha||=1,\alpha\Sigma^\top\varphi_{l}=0,l=1,\dots,m-1
\end{align} $$
Inflates directions of high variance, but shrinks low variance like ridge regression.

PLS maximizes the *covariance* between $(Xu, Yv)$, whereas PCA maximizes *internal variance*.

>[!faq] Example PLS vs PCA
>>[!faq]+ First Example
>>Let X be wearable biosignals (Heart Rate, Actigraphy) and y be OCD severity.
>>
>>The highest variance in X (PCA) might just be the child running or playing sports. PLS ignores this movement noise and finds the hidden subspace of physiological signals that directly correlate with an OCD episode.
>
>>[!faq]+ Second Example
>>In a Speech Emotion task (y = anger level): 
>>**PCR Failure:** PCA keeps the loud background hum (high variance) but drops a subtle, high-frequency vocal quiver (low variance). PCR fails to predict anger. 
>>**PLS Success:** PLS forces the subspace to look for features that explain y. It elevates the subtle quiver because Corr2(y, Xα) is very high. 

# Canonical Correlation Analysis
Find *associates* between two data groups $X$ and $Y$, given as
$$ \max_{u_{m},v_{m}}\text{Corr}^2(Yu_{m},Xv_{m}), $$
subject to $$ u_{m}u_{j}=0,v_{m}v_{j}=0,m\neq j,m=1,\dots, m $$
The linear combinations are now uncorrelated and there is at most a minimum dimension of the two data matrices.

We seek *canonical* variates: $U=Xu,V=Yv$