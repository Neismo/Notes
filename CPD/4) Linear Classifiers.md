
For classification, we want to know the *posterior*, i.e., $p(G=k|X)$, the probability of class $k$ given $X$, for optimal classification.

If we denote $f_k(x)$ as the class-conditional density of X in class $G=k$, and denote the prior probability for class $k$ as $\pi_k$, the a simple application of Bayes gives us:

$$ p(G=k|X=x) = \frac{p(X=x|G=k)p(G=k)}{\sum_{i=1}^Kp(X=x|G=i)p(G=i)} = \frac{f_{k}(x)\pi_{k}}{\sum_{i=1}^Kf_{i}(x)\pi_{i}} \propto f_{k}(x)\pi_{k} $$

We see that in order to *classify*, we mostly require the term $f_{k}(x)$. For that we have:
- Linear and Quadratic Discriminant Analysis which use ==Gaussian densities==.
- Mixtures of Gaussian (MoG) allow for ==Non-linear Boundaries==.
- General non-parametric models allows each density to be estimated, more ==flexible==.
- *Naive Bayes'* models are a simple model that assumes inputs are ==conditionally independent== in each class!

# Linear Discriminant Analysis
In this case we model the class densities as multivariate Gaussian distributions (assumption):
$$ f_{k}(x)=\frac{1}{(2\pi)^{p/2}|\Sigma_{k}|^{1/2}}\exp\left( -\frac{1}{2}(x-\mathbf{\mu}_{k})^\top\Sigma_{k}^{-1}(x-\mathbf{\mu}_{k}) \right), $$
where it is assumes that each class ==share covariance matrix==: $\mathbf{\Sigma}_{k}=\mathbf{\Sigma}\, \forall k$. When comparing two classes, it is sufficient to do it in *log* space, and so we take the log fraction:
$$ \begin{align}
\log \frac{p(G=k|X=x)}{p(G=l|X=x)} &= \log \frac{f_{k}(x)}{f_{l}(x)}+\log \frac{\pi_{k}}{\pi_{l}} \\
&= \log \frac{\pi_{k}}{\pi_{l}} -\frac{1}{2}(\mu_{k}+\mu_{l})^\top\mathbf{\Sigma}^{-1}(\mu_{k}-\mu_{l})+x^\top\mathbf{\Sigma}^{-1}(\mu_{k}-\mu_{l}) \tag{1},
\end{align} $$
We can also write out the ==linear discriminant functions==:
$$ \delta_{k}(x) = x^\top\mathbf{\Sigma}^{-1}\mu_{k}-\frac{1}{2}\mu_{k}^\top\mathbf{\Sigma}^{-1}\mu_{k}+\log \pi_{k},\tag{2}$$

such that we can write (1) as:
$$ \log \frac{p(G=k|X=x)}{p(G=l|X=x)} = \delta_{k}(x)-\delta_{l}(x) $$

because the covariance is equal among classes, the quadratic expression is cancelled out, and this is why it is *linear*. This means all decision boundaries will be *linear* in the *p*-dimension.

In practice, we must estimate the Gaussian parameters with training data:
- $\hat{\pi}_{k}=\frac{N_{k}}{N}$, where $N_{k}$ is the number of training points belonging to class $k$
- $\hat{\mu}_{k}=\sum_{g_{i}=k}\frac{x_{i}}{N_{k}}$
- $\hat{\mathbf{\Sigma}}=\frac{\sum_{k=1}^K\sum_{g_{i}=k}(x_{i}-\hat{\mu}_{k})(x_{i}-\hat{\mu}_{k})^\top}{(N-K)}$


**Classification Rule**: $\hat{G}_{k}=\arg\max_{k}\delta_{k}(x)$

## Quadratic Discriminant Analysis
In equation (1) we assumed a shared covariance matrix between the classes. If we do not assume this, the quadratic term ==does not cancel out== in (1), and we get the *quadratic linear function*:

$$ \delta_{k}(x)=-\frac{1}{2}\log|\mathbf{\Sigma}_{k}|-\frac{1}{2}(x-\mu_{k})^\top\mathbf{\Sigma}_{k}^{-1}(x-\mu_{k})+\log \pi_{k}, \tag{3} $$

When $p$ is large, the amount of extra computations can be quite dramatic, as we must estimate the covariance for each class. 

LDA requires $(K-1)(p-1)$ parameters, but QDA requires $(K-1)\left( \frac{p(p+3)}{2}+1 \right)$.

## Regularized LDA
Standard LDA or QDA ==fails in high-dimensional data==, with low data points, due to the covariance matrix inversion. It is possible to do a compromise between the two by allowing a $\alpha$ parameter:

$$ \hat{\Sigma}_{k}(\alpha)=\alpha \hat{\mathbf{\Sigma}}_{k}+(1-\alpha)\hat{\mathbf{\Sigma}}, \tag{4} $$

where $\hat{\mathbf{\Sigma}}$ is the pooled covariance from LDA, and $\alpha \in[0,1]$. Usually chosen through cross-validation. This can help save the model in those cases.

### Other Options:

1) **Shrink towards diagonal**: $\hat{\Sigma}(\gamma)=\gamma\hat{\Sigma}+(1-\gamma)\hat{\Sigma}$

# Logistic Regression
We want linear functions and sum probabilities to one - no longer an assumption on data distribution:
$$\begin{align}
\log \frac{p(G=1|X=x)}{p(G=K|X=x)} &= \beta_{10}+\beta_{1}^\top x \\
\log \frac{p(G=2|X=x)}{p(G=K|X=x)} &= \beta_{20}+\beta_{2}^\top x \\
\cdots \\
\log \frac{p(G=K-1|X=x)}{p(G=K|X=x)} &= \beta_{(K-1)0}+\beta_{K-1}^\top x,
\end{align}$$

In this case, the last class is used as shared denominator for odd-ratios, the choice is arbitrary ==(?)==. Let us denote the probability of $p(G=k|X=x)$ as $p_{k}(x;\theta)$, parameterized by $\theta=\{ \beta_{10},\beta_{1}^\top, \beta_{20}, \beta_{2}^\top,\cdots,\beta_{(K-1)0},\beta_{K-1}^\top\}$.

We use ***Sigmoid*** functions for two-class problems:
$$\begin{align}
p(G=1|X=x) &= \frac{\exp(\beta_{10}+\beta_{1}^\top x)}{1+\exp(\beta_{10}+\beta_{1}^\top x)} \\
p(G=2|X=x) &= \frac{1}{1+\exp(\beta_{10}+\beta_{1}^\top x)}
\end{align}$$

and ***softmax*** for multi-class problems.

## Fitting Logistic Regression
Usually fitted with *maximum likelihood* (MLE). The log-likelihood for $N$ observations are, and using the two-class case, such that $p_{1}(x|\beta)=p(x|\beta)$, and $p_{2}(x|\beta)=(1-p(x|\beta))$, and $y_{i}=1$ when class is 1, otherwise $y_{i}=0$

$$ \begin{align}
\mathcal{L}(\beta) &= \sum_{i=1}^N\{y_{i}\log p(x_{i}|\beta) + (1-y_{i})\log(1-p(x_{i}|\beta))\} \\
&= \sum_{i=1}^N\{y_{i}\beta^\top x_{i} - \log(1+\exp(\beta^\top x_{i}))\}, \tag{5}
\end{align} $$

now, we assume $\beta=\{\beta_{10},\beta_{1^\top}\},$ and we assume the vector $x_i$ also has a $1$ column for the intercept. So we set the derivative of (5) to 0, and solve:

$$\begin{align}
\frac{\partial\mathcal{L}(\beta)}{\partial \beta}=\sum_{i=1}^Nx_{i}(y_{i}-p(x_{i}|\beta)) = 0,
\end{align}$$

which are $p+1$ equations non-linear in $\beta$. If the *hessian* is also calculated:
$$ \frac{\partial^2\mathcal{L}(\beta)}{\partial^2\beta} = -\sum_{i=1}^Nx_{i}x_{i}^\top p(x_{i}|\beta)(1-p(x_{i}|\beta)), $$

then we can solve the equations of (5) with **Iterative Least Squares Method**, with a starting guess $\beta^{old}$:
$$ \beta^{new} = \beta^{old} - \left(\frac{\partial^2\mathcal{L}(\beta)}{\partial^2\beta}\right)^{-1}\cdot \frac{\partial\mathcal{L}(\beta)}{\partial \beta}.$$

It is sometimes more convenient to write in *matrix notation*: let $\mathbf{y}$ denote the response vector of $y_i$, $\mathbf{X}$ as the $(N \times(p+1))$ matrix, $\mathbf{p}$ the vector of *fitted* probabilities, and $\mathbf{W}$ as the $N\times N$ diagonal matrix of weights with the *i*th diagonal element $p(x_{i}|\beta^{old})(1-p(x_{i}|\beta^{old}))$:
$$\begin{align}
 \frac{\partial^2\mathcal{L}(\beta)}{\partial^2\beta} &= \mathbf{X}^\top(\mathbf{y}-\mathbf{p}) \\
   \frac{\partial^2\mathcal{L}(\beta)}{\partial^2\beta} &= -\mathbf{X}^\top\mathbf{W}\mathbf{X}, 
\end{align}$$

and the update step becomes
$$\begin{align}
\beta^{new} &= \beta^{old} + (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top(\mathbf{y}-\mathbf{p}) \\
&= (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}(\mathbf{X}\beta^{old}+\mathbf{W}^{-1}(\mathbf{y}-\mathbf{p})) \\
&= (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}\mathbf{z},
\end{align}$$

where $\mathbf{z}=(\mathbf{X}\beta^{old}+\mathbf{W}^{-1}(\mathbf{y}-\mathbf{p}))$. At each iteration $\mathbf{p}$ changes and so does $\mathbf{W}$ and $\mathbf{z}$. Convergence is never guaranteed, and $\beta = \mathbf{0}$ is a good starting guess.

## Properties:
- More robust than LDA as fewer assumptions
- Handles categorical variables better than LDA
- Observations far away from boundary are *down-weighted*
- Breaks down when perfectly seperable
- Easy to interpret and explain
- Can be regularized
- Generalized to multi-class problem (softmax)

# LDA vs Logistic Regression
LDA:
- Assumes features are Gaussian.
- Uses all data points to estimate mean and covariance.
- More stable when assumptions hold, especially for small N.

Logistic Regression:
- No assumptions about data distribution
- Focuses primarily on data points near decision boundary
- Generally safer and more robust.

![[ModelsComparison.png]]

# Basis Expansion & Splines

Sometimes we want non-linear realities, but above are linear models. To do this, we can transform the input with various basis functions, and concatenate these to our data matrix.

**Idea**: replace variables (columns) of our matrix $\mathbf{X}$, with transformations $h(\mathbf{X})$.

## Polynomials & Step Functions
==Polynomial basis expansion== has a flaw in that they are *global*; tweaking a high-degree polynomial will mean they extrapolate very differently, usually *very* poorly. 

==Step functions (binning)== are *discontinuous* in their separating regions. 

**Solution**: *piece-wise polynomials*, so a *combination*! Put the **knots**(separation regions) at place of statistical interest. For example, if some disease is observed when Y > 123, then set knot at 123.

Use up to $x^3$, no more needed as that will give *smooth* enough curves between the knots (derivative and hessian).

![[SplineApproximation.png]]

Here we denote $(x-\delta)_{+}^3$ as the **hinge function**, which is defined as:
- If $x<\delta$, the value is $0$!
- If $x>=\delta$, then it adds a new cubic trajectory from the data. Stats at 0 at $\delta$.