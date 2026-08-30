Higher moments in Pareto have much heavier tails; problem when calculating variance, means etc.
From exercise yesterday, we saw variance was not stable about the theoretical ones; this is because a lot of the mass for the Pareto is early; while for variance with its heavy tails it is much less. We need variance reduction.
- Large samples do not guarantee accurate estimates.
- Confidence intervals can be misleading if regions of the distribution are only sampled rarely.
- Rare observations may contribute disproportionately to moments.

## Variance Reduction
Main idea is to exploit:
- properties of a model
- correlation between variables
- Structural knowledge of sample space

Methods considered:
- Antithetic variables
- Control variate
- Stratified sampling
- Importance sampling
- Common random numbers

 Suppose $\theta = \mathbb{E}[X]$, and its Monte Carlo estimator $\hat{\theta}$. The precision of the estimator is determined by its variance:
$$ \mathbb{V}[\hat\theta]=\mathbb{V}[X_i]/n $$
which means we can reduce it by increasing $n$. **Alternative**, as can be seen in numerator, can we reduce the variance of the samples.

**Example**: evaluate $\theta=\int_0^1\exp(x)dx$. Introducing $U\sim U(0,1)$ it is $\theta = \int_0^1\exp(x) \cdot1dx = \mathbb{E}[\exp(U)]$.
Monte Carlo is then: $\frac{1}{n}\sum_i^n\exp(U_i)$.

Benchmark variance of the observations $X_i = \exp(U_i)$ is: $0.2420$; variance reduction methods should reduce this!

**Antithetic Variables**: use counterpart with negative correlation of samples to reduce variance. From above example, with $U_i$ sample also $1-U_i$ and take the average; $Y_i = \frac{\exp(U_i) + \exp(1-U_i)}{2}$.

This new variable has the same mean. Variance is now $\mathbb{V}[Y_i] = \frac{1}{2}(0.2420 - 0.2342) = 0.0039$. Much lower. Not much more computationally expensive, we use the same random number, to generate its antithetic.

Generally, it work when $X_i$ also works on several variables, as long as each coordinate is monotone. In queue system: increasing serving time, blocking increases.

*In queue system exercise*: $\theta = P(m)$, $\hat\theta = \frac{1}{n}\sum_{i=1}^n B_i$, where $B_i$ is 1 if blocked, 0 if enter.
$\mathbb{E}[B_i] = P(m)=\theta$. Can we use antithetic in system? We need to formulate $B_i$ as a formula of service time before.

$B_i (S_1,S_2,\dots,S_{i-1},A_1,A_2,\dots,A_{i-1})$. In all these coordinates (service time of previous, inter-arrival time) it is monotone in eaB_ich of the variables. $S_i\sim\exp(\lambda), A_i\sim\exp(1/\mu)$. which we generated with $U_i$. So idea is to have a second "replica" simulation where we use the same $U_i$ but with instead $1 - U_i$.

So now use $\frac{B_i+\hat B_i}{2}$.

**Control Variate**: exploit dependence between variables:
$$ Y_i = X_i + c(Z_i - \mu_Z) $$
where we must know the mean of $Z_i$ following a distribution we know. This ensure the mean is the same of $X_i$.

The variance will be $$ V[Y_i] = V[X_i] + c^2V[Z_i] + 2cCov(X_i,Y_i), $$
where
$$ c = -\frac{Cov(X_i,Z_i)}{V[Z_i]}, $$
which gives a variance:
$$ V[Y_i] = V[X_i]-\frac{Cov(X_i,Z_i)^2}{V[Z_i]}. $$
Example in slides using $U\sim U(0,1)$.

**Stratified Sampling**: 
Divide samples space into $m$ regions, sample from each of them, this is your new observation as an average: $$ Y_i = \frac{1}{m}\sum_{j=1}^m X_{ij} $$
**Importance Sampling**:

