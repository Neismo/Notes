
# Random Forests
Define number of trees $B$; Typically >100

1) for $b=1$ to $B$ do:
	1) Take a random sample of size $N$ with replacement (bootstrap) from data $p$.
	2) Repeat following until tree reaches minimum node size on it (no pruning):
		1) Take a random sample without replacement; i.e., subset of the bootstrapped data $m<p$
		2) Construct the first CART partition of the data based on this $m$;
	3) Validate on OOB (out of bag) samples
2) Output B trees.

**Classification**: majority votes
**Regression**: Prediction is average of all B trees $\hat{y} = \frac{1}{B}\sum_{b=1}^BT_{b}(\mathbf{x})$.

**Out-of-bag estimates**: samples not in the bootstrap can be used for fair validation of each tree

**How many trees**; stop when error based on OOB estimates is decreasing 

**Choosing $\mathbf{m}$**:
- Classification: $m=\text{floor}(\sqrt{ p })$
- Regression: $m=\text{floor}\left( \frac{p}{3} \right)$

RF also works when $p>n$; i.e., more variables than observations

## Variable Importance

Two approaches
- **GINI index**
	- Improvement in the split criterion at each split is accumulated over all the trees for each variable
- **OOB estimates**
	- Measure prediction strength by dropping OOB samples through the trees; then do again but permute variable $j$; An *average* of the *difference in accuracy* for all the trees gives the measure of variable $j$.

**Proximity plots**: look at OOB observations in a 2D plot (for two variables); if OOB observations end up in same place, then add 1 to the value of that place in the plot.

# Boosting

Average many trees; but adds a *weighting* to them. Boosting **reduces bias**; favors many *small trees*; reduces *both* *variance* and *bias*

## AdaBoost
![[Pasted image 20260312090707 1.png]]

1) Initialize weights as $w_{i}=\frac{1}{N}$.
2) For $m=1$ to $M$ do:
	1) Fit classifier $G_{m}(x)$ to training data with the weights
	2) Computed weighted error
	$$ \text{err}_{m} $$
	3) Compute $\alpha_{m}=\log\left[ \frac{1-\text{err}_{m}}{\text{err}_{m}} \right]$
	4) Update weights


Committee of *weak learners*. Dominates *bagging*. Can have a *learning rate*.

Problem when *incorrectly labeled* data exists; will try and push more weight on those to classify them.

## Boosting & Additive models
Each classifier can be seen as as a basis function
$$ G(x) = \text{sign}\left[ \sum_{m=1}^M \alpha_{m}G_{m}(x)\right] $$
and so expressed as
$$F(x)=\sum_{m=1}^M\beta_{m}b(x,\gamma_{m}),$$
where $\gamma_m$ and  $\beta_m$ are learned. With MLE or similar we jointly find them. With boosting, we find one at a time. 

### Forward Stagewise Additive Modelling
![[CART.png]]

# Exercises

## Derive the variance of the average of $B$ i.i.d random variables with variance $\sigma^2$.
$$\begin{align}
\mathbb{V}\left[ \frac{1}{B}\sum_{i=1}^BB_{i} \right] &= \frac{1}{B^2}\mathbb{V}\left[ \sum_{i=1}^BB_{i} \right] \\
&= \frac{1}{B^2}B\sigma^2 \\
&= \frac{\sigma^2}{B},
\end{align}$$
and above tends to $0$ as $B\to \infty$.

## Derive the variance of the average of B i.d. but dependent random variables with variance $\sigma^2$ and correlation $\rho$.

$$ \begin{align}
\mathbb{V}\left[ \frac{1}{B}\sum_{i=1}^BB_{i} \right] &= \frac{1}{B^2}\mathbb{V}\left[ \sum_{i=1}^BB_{i} \right] \\
&= \frac{1}{B^2}\left(\sum_{i=1}^B\sum_{j=1}^B\text{Cov}(B_{i},B_{j})\right) \\
&= \frac{1}{B^2}\left(\sum_{i=1}^B \mathbb{V}[B_{i}] + \sum_{i=1}^B\sum_{j=1,j\neq i}^B\text{Cov}(B_{i},B_{j})\right) \\
&= \frac{1}{B^2}\left( \sum_{i=1}^B\sigma^2 + \sum_{i=1}^B\sum_{j=1,j\neq i}^B\rho\sigma^2 \right) \\
&= \frac{1}{B^2}(B\sigma^2 + (B^2-B)\rho \sigma^2) \\
&= \frac{B\sigma^2 + (B^2-B)\rho \sigma^2}{B^2} \\
&= \frac{B\sigma^2 + B(B-1)\rho \sigma^2}{B^2} \\
&= \frac{\sigma^2 + (B-1)\rho \sigma^2}{B} \\
&= \frac{\sigma^2}{B} + \frac{B\rho \sigma^2}{B} - \frac{\rho\sigma^2}{B} \\
&= \frac{\sigma^2}{B} - \frac{\rho\sigma^2}{B} + \rho \sigma^2 \\
&= \frac{1-\rho}{B}\sigma^2 + \rho \sigma^2
\end{align} $$
## Relation of above to plots below
![[TreeCorrelation.png]]
This shows that when we do random forests and split on a smaller subset of the data at each split, we reduce the variance of the bagged trees we couldn't get with just bagged trees.