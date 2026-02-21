
Used when the number of features are similar to the number of data points. ($p\sim N$). Sometimes called ==shrinkage methods==, and *Ridge Regression* is one such method (it does not zero parameters).

# Curse of Dimensionality

When the number of dimension grow, i.e., the ==solution space==, the computational complexity grows, usually in orders $O(N^2)\geq$.

Problems:
- **Sparsity**: some (local) regions become super sparse/empty
- **Distances**: lose meaning; points can become *roughly* equidistant
- **Overfitting**: We can sometimes *perfectly fit noise*.
- **Edge Effect**: Most points reside at *boundaries*
- **Computational Costs**: search algorithms can slow down

### "Blessings" of dimensionality
- Several features can be correlated, and we can *average* over them.
- Underlying distribution will be finite, and will usually lie on a lower-dimensionality manifold.
- Underlying structure in data, will give an approximate finite dimensionality.

# Dimensionality Reduction
Decreasing dimension, and/or identifying the important features.
$$ L_{2}=||\beta||_{2}^2=\sqrt{\sum_{k}a_{k}^2 }^2 = \sum_{k}a_{k}^2,\quad L_{1}=\sum_{k}|a_{k}| $$

## Shrinkage Methods

Instead of zero-ing out a subset of coefficients, we can **shrink** instead; *ridge* with $L_{2}$, *Lasso* with $L_1,$ *Elastic Net* which is a *hybrid*.

### Ridge Regression
*Shrinks*, but doesn't set any parameters to 0. The intercept, $\beta_{0}$, is usually not penalized. Optimal model parameters are easily computed. Will also have a *grouping effect*!

### Lasso
Similar to Ridge Regression, but uses $L_{1}$ penalty:
- No closed form solution, requires quadratic programming to solve
- For large $\lambda$'s, some parameters *will* be set to 0.

Usually hits an axis for the Lasso-parameter, which means it will be set to 0.

To solve Lasso, one can do these two at least:
- Least Angle Regression Selection (LARS)
- Cyclical Coordinate Descent

Lasso does *not really have a grouping effect*.
#### LARS
Used for finding Lasso/Elastic Net paths

Algorithm:
1) Start with $\beta=0$, find variable $x_{j}$ most correlated with $y$.
2) Move $\beta_{j}$ in the direction of its least-squares coefficient.
3) Stop when another variable $x_{k}$ has as much correlation with $y$ and $x_{k}$.
4) Move in a direction *equiangular* to both $x_{j},x_{k}$!

Modifications:
- If some parameter crosses the zero, set it to zero, and re-compute. More stable!

![[LassoVsLassoModified.png]]Question is, *which iteration do we stop at*? Use **cross-validation**, it is model selection after all. We can also choose a point for it with a $c_{p}$ like statistic, which is *not reliable* for when $p\gg n$!:
$$ C_{p} = \frac{1}{\hat{\sigma}^2}\sum_{i=1}^n (y_{i}-\hat{y}_{i})^2-n+2k,$$
where $k$ is number of steps. The reason it does not work well when $p\gg n$, is because the noise $\hat{\sigma}^2$ will be *veeery low* due to *overfitting*!

Three shortcomings:
1) **High Dimensionality**, in $p>n$ case it select at most $n$ variables
2) **Grouping Effect**: Group of variables with high correlation, LASSO selects one arbitrarily.
3) **Predictive Power**: $n>p$: 

### Cyclical
Assumes standardized data. Here we *fix* $\lambda$, and solve:
$$ \min_{\beta}\frac{1}{2n}\sum_{i=1}^n (y_{i}-x_{i}\beta)^2+\lambda|\beta|, $$
iteratively by updating one coordinate $\beta_{j}$ at a time, while holding the others fixed in the current estimate $\hat{\beta}_{k}$. 

### Elastic Net
Combination of LASSO and Ridge regression:
$$ \min_{\beta}\frac{1}{2n}||Y-X\beta||^2_{2} + \lambda\left( \frac{1}{2}(1-\alpha)||\beta||_{2}^2+\alpha||\beta||_{1} \right) $$
So:
- $\alpha=1$: LASSO
- $\alpha=0$: Ridge
- $0<\alpha<1$: The *elastic* region

#### Computations
To solve this, we *hide* the $L_{2}$ penalty *inside the data*:
$$ X^*_{(n+m),m} = \begin{pmatrix} X \\
\sqrt{ \lambda_{2} }\mathbf{I}_{m}
\end{pmatrix},\quad y^*_{(n+m)} = \begin{pmatrix}
y \\
\mathbf{0}_{m}
\end{pmatrix}$$
Which means we end up with a plain LASSO problem we can solve! 

## Combinatoric Search
*Idea*: try all combinations of features, and select the optimal one.
- Will find the best solution, but can be super *expensive*!

**Forward Selection**:
- Reasonable number of models to test, used when $p>n$, but might not give optimal selection set

**Backwards Selection**:
- Reasonable number of models to test, and "usually" better than forward selection; still might not give optimal selection set.

# Multiple Testing

We can do statistical tests for if parameters are different from each other (t-test), or zero (F-test). If we do **one** test, then we have $\alpha$ chance of falsely rejecting the hypothesis. But if do more with $\alpha$ level tests, then that is no longer the case!

**Family-Wise Error Rate (FWER)**: $$ FWER = 1-(1-\alpha)^M, $$
for $M$ tests.

## Bonferroni Correction

Reject only now if $p$-value is $<\frac{\alpha}{M}$! This ensure the chance of falsely rejecting a hypothesis to be $\alpha$ again! It does result in *lower power* though.

**False Discovery Rate** (or False Positive Rate):
$$ FDR = E\left( \frac{FP}{FP+TP} \right), $$
where $E$ is ???. We accept hypothesis where $FDR<q$, and we expect $q$ mistakes.

## Benjamini-Hochbergs
Assume $m$ tests, and $m$ p-values: $p_{(1)},\cdots p_{(m)}$. Sort by value in ascending order,

For a given $q$ find $k = \max \left\{ i:p_{(i)}\leq \frac{i}{m}q \right\}$, and reject $H_{(1),\cdots,H_{(k)}}$.