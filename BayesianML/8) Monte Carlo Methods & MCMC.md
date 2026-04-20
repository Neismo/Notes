
# More on Calibration

Recall the ECE
$$  \text{ECE} = \sum_{b=1}^B \frac{|\mathcal{B}_{b}|}{N}|\text{acc}(\mathcal{B}_{b}) - \text{conf}(\mathcal{B}_{b})|  $$
where the constant $|\mathcal{B}_{b}|$ is a scaling based on the size of the bin. ==What about regression models==? What's the "binning" we should do?

Recall the CDF for the predictive density $p(y^*|\mathbf{y},\mathbf{x}^*) \equiv p(y^*)$ (not the prior here!)
$$ F(\tau) \equiv P(y^* \leq \tau)=\int_{-\infty}^\tau p(y^*)\text{d}y^*\in[0,1]$$
We are in a sense "discretizing" the CDF of our predictive density, and using that:
$$ \hat{p} = \frac{\sum_{n=1}^N\mathbb{I}[y^*_{n}\leq F^{-1}_{n}(p)]}{N} \underbrace{\to}_{\text{Calibrated}} p,$$
where $F_{n}$ is the CDF of the predictive distribution for the $n$'th datapoint. $p$ would run over the unit interval again. We in practice check each test evaluation if it is inside the CDF up to the point *p*. This can extend to ==intervals as well== $p_{1},p_{2}\in[0,1]$:
$$ \frac{\sum_{n=1}^N\mathbb{I}[F_{n}^{-1}(p_{1})\leq y^*_{n}\leq F^{-1}_{n}(p_{2})]}{N} \underbrace{\to}_{\text{Calibrated}} p_{2}-p_{1},$$
So, to sum up; we need the CDF for each predictive distribution to do this; a bit more *abstract* than for classification.
# Bayesian Methods & Annoying Integrals

As always, when we compute the posterior distribution, we use Bayes'
$$ p(\mathbf{w}|\mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{w})p(\mathbf{w})}{p(\mathbf{y})}, $$
and when making predictions we evaluate the posterior predictive distribution
$$ p(y^*|\mathbf{y},\mathbf{x}^*) = \int p(y^*|\mathbf{y},\mathbf{x^*})p(\mathbf{w}|\mathbf{y})\text{d}\mathbf{w} $$
Both require usually intractable integrals, and *conjugate priors* is a neat trick but not always available.

For general Bayesian Inference we rely on
- Laplace Approximations
- Variational Approximations
- ==Markov Chain Monte Carlo Methods==

## Monte Carlo

Methods that rely on ==random sampling==! It breaks the *curse of dimensionality*. It is a general tool for approximating expectations
$$ \mathbb{E}_{p}[f(\mathbf{z})] = \int p(\mathbf{z})f(\mathbf{z}) \text{d}\mathbf{z}. $$

That means if we can formulate desired properties on expectation form, we can use monte carlo. Here are some examples:
$$ \begin{align}
\mathbb{E}[\mathbf{w}] = \int \mathbf{w}p(\mathbf{w}|\mathbf{y})\text{d}\mathbf{w}&=\mathbb{E}[f(\mathbf{w})],\quad \text{for } f(\mathbf{w})=\mathbf{w} \tag{Mean} \\
\mathbb{V}[\mathbf{w}] = \int (\mathbf{w}- \mathbb{E}[\mathbf{w}])^2p(\mathbf{w}|\mathbf{y})\text{d}\mathbf{w}&=\mathbb{E}[f(\mathbf{w})],\quad \text{for } f(\mathbf{w})=(\mathbf{w} - \mathbb{E}[\mathbf{w}])^2 \tag{Variance} \\
\mathbb{P}[w>\tau] = \int_{\tau}^\infty p(w|\mathbf{y})\text{d}w&=\mathbb{E}[f(w)],\quad \text{for } f(w)=\mathbb{I}[w>\tau] \tag{Probs} \\
p(y^*|\mathbf{y},\mathbf{x}^*) = \int p(y^*|\mathbf{w,x}^*)p(\mathbf{w}|\mathbf{y})\text{d}\mathbf{w}&=\mathbb{E}[f(w)],\quad \text{for } f(w)=p(y^*|\mathbf{y,w}^*), \tag{Preds}
\end{align} $$
And as can be seen, most are able to be re-written as expectations!

### Monte Carlo Integration
Say we now want to evaluate the expectation
$$ \bar{f} = \mathbb{E}[f(\mathbf{z})] = \int p(\mathbf{z})f(\mathbf{z})\text{d}\mathbf{z}. $$

If we can sample $S$ ==i.i.d samples== from $p(\mathbf{z})$, then the ==Monte Carlo estimator== is given by:
$$ \bar{f}\approx \hat{f} =\frac{1}{S}\sum_{i=1}^Sf(\mathbf{z}^i) $$

If you look at the expectation we arrive at an *unbiased estimator*
$$ \mathbb{E}[\hat{f}] = \mathbb{E}\left[ \frac{1}{S}\sum_{1}^Sf(\mathbf{z}^i) \right] = \frac{1}{S}\sum_{i=1}^S\mathbb{E}[f(\mathbf{z}^i)] = \frac{1}{S}\sum_{i=1}^S \bar{f} = \bar{f}. $$
If you do the same for variance you get $\mathbb{V}[\hat{f}] = \frac{1}{S}\mathbb{V}[f(\mathbf{z})]$, which means we can simply increase the number of samples to get a more accurate result!

The **Monte Carlo Standard Error (MCSE)** is the expected difference between $\hat{f}$ and $\bar{f}$:
$$ \text{MCSE}_{f} = \frac{1}{\sqrt{ S }}\sqrt{ \mathbb{V}[f(\mathbf{z})] }. $$

This is the number we report for example $\hat{f} \pm \text{MCSE}_{f}$!

# Simple Sampling Methods

## Ancestral Sampling
Useful for *sampling* and *joint* distributions; consider the linear model seen in week 3 
$$ p(\mathbf{y},\mathbf{w}|\kappa^2,\sigma^2) = \mathcal{N}(\mathbf{y}|\mathbf{Xw},\sigma^2\mathbf{I})\mathcal{N}(\mathbf{w}|\mathbf{0},\kappa^2\mathbf{I}). $$

Here we optimized the parameters with $\kappa^2=\alpha^{-1}$ and $\sigma^2=\beta^{-1}$ using the marginalized likelihood. What if want ==hyperpriors== on $\kappa,\sigma$ as well? For example ==half-normals==:
$$ \begin{align}
p(\kappa^2) = \mathcal{N}_{+}(\kappa^2|0,1) \\
p(\sigma^2) = \mathcal{N}_{+}(\sigma^2|0,1)
\end{align} $$
Then the joint model becomes
$$  p(\mathbf{y},\mathbf{w},\kappa^2,\sigma^2) = p(\mathbf{y}|\mathbf{w},\sigma^2)p(\mathbf{w}|\kappa^2)p(\sigma^2)p(\kappa^2). $$

We can generate samples from the join distribution to reason about it and understand the new model. Called ==ancestral sampling==. 

SEE EXAMPLE IN WEEK 8 SLIDES; idea is to generate samples from the priors, see how the likelihood changes with them; then compare if we change the distribution of the priors!

## Rejection Sampling
A simple tool for generating samples from a distribution $p(\mathbf{z})$.
$$ p(\mathbf{z}) = \frac{1}{Z}\tilde{p}(\mathbf{z}) $$
Let $q(\mathbf{z})$ be an "easy", but similar, distribution (normal distributions or similar) and $K>0$  a constant such that
$$ \tilde{p}(\mathbf{z}) \leq Kq(\mathbf{z}),\text{ for all }\mathbf{z}. $$
So we are looking for a distribution that "traps" the one we are interested in inside it.

Steps for sampling is then:
1) Sample $\mathbf{z} \sim q(\mathbf{z})$
2) Sample $u|\mathbf{z}$ from a uniform distribution $u\sim \mathcal{U}[0,Kq(\mathbf{z})]$.
3) If $u>\tilde{p}(\mathbf{z})$ reject the sample, otherwise keep!

It is very easy to use and works well in lower dimensions, but in high dimensions we can see *low acceptance rates*! Also requires finding the K.

![[RejectionSampling.png]]

## Importance Sampling
Not a technique for generating samples, but for estimating the expectation of a distribution, when we can't sample directly! We again make use of a "easy" distribution.

$$ \begin{align}
\mathbb{E}_{p}[f(\mathbf{z})] &= \int f(\mathbf{z})p(\mathbf{z}) \text{d}\mathbf{z} \\
&= \int f(\mathbf{z}) \underbrace{\frac{q(\mathbf{z})}{q(\mathbf{z})}}_{=1} p(\mathbf{z})\text{d}\mathbf{z} \\
&= \int f(\mathbf{z}) \frac{p(\mathbf{z})}{q(\mathbf{z})}q(\mathbf{z})\text{d}\mathbf{z} \\
&= \mathbb{E}_{q}\left[ f(\mathbf{z})\frac{p(\mathbf{z})}{q(\mathbf{z})} \right] \\
&\approx \frac{1}{S}\sum_{i=1}^S f(\mathbf{z}^i)\frac{p(\mathbf{z}^i)}{q(\mathbf{z}^i)} \equiv \hat{f}, \text{ for } \mathbf{z}^{i}\sim q(\mathbf{z}).
\end{align} $$

Where we used a neat trick of inserting $\frac{q(\mathbf{z})}{q(\mathbf{z})}$. The ratios $\frac{p(\mathbf{z}^i)}{q(\mathbf{z}^i)}$ are called *importance weights*. The make up for the differences we couldn't model with the easy distribution. And $q$ could be a Laplace approximation of $p$!
# Markov Chains Monte Carlo (MCMC) Methods

Turns out it is hard to draw I.I.D samples from arbitrary distributions, that is where MCMC methods come into play!

Idea is to "give up" on I.I.D and instead do random walks with desired properties.

## Markov Chains
A first-order Markov chain is defined as a series of random variables $\mathbf{z}^{(1)},\mathbf{z}^{(2)},\cdots,\mathbf{z}^{(m)}$ with the following property
$$ p\left(\mathbf{z}^{(m+1)}|\mathbf{z}^{(1)},\mathbf{z}^{(1)},\cdots,\mathbf{z}^{(m)}\right) = p\left(\mathbf{z}^{(m+1)}|\mathbf{z}^{(m)}\right). $$
Implications of this for a joint distribution
$$ p(\mathbf{z}_{1},\mathbf{z}_{2},\mathbf{z}_{3},\mathbf{z}_{4}) = p(\mathbf{z}_{4}|\mathbf{z}_{3},\mathbf{z}_{2},\mathbf{z}_{1})p(\mathbf{z}_{3}|\mathbf{z}_{2},\mathbf{z}_{1})p(\mathbf{z}_{2}|\mathbf{z}_{1})p(\mathbf{z}_{1}) = p(\mathbf{z}_{4}|\mathbf{z}_{3})p(\mathbf{z}_{3}|\mathbf{z}_{2})p(\mathbf{z}_{2}|\mathbf{z}_{1})p(\mathbf{z}_{1}). $$
So it is much simpler! We can formulate the ==transition kernel== which defined how to move between time steps, which *may* depend on *m*. It is ==*homogeneous* if it is the same for *all m*==!
$$ T_{m}(\mathbf{z}^{(m)},\mathbf{z}^{(m+1)}) \equiv p (\mathbf{z}^{(m+1)}|\mathbf{z}^{(m)}) $$

The distribution at some time step $m+1$ is calculated with the sum rule:
$$ p(\mathbf{z}^{(m+1)}) = \int p(\mathbf{z}^{(m+1)}|\mathbf{z}^{(m)})p(\mathbf{z}^{(m)})\text{d}\mathbf{z}^{(m)}. $$

A distribution is said to be ==stationary== or ==invariant== wrt. the Markov chain if the distribution does not change with any steps (see how it is the same $p^*$ below)
$$ p^*(\mathbf{z}) = \int T(\mathbf{z}',\mathbf{z})p^*(\mathbf{z}') \text{d}\mathbf{z}'. $$

**The idea**: form a Markov chain such that the target distribution is invariant wrt the chain! That is, design a very specific walk where the random walk spends most time in high density regions!

### Metropolis Algorithms
Gives recipe for designing the above idea! Suppose the goal is to sample from a target distribution $p(\theta)$, and assume we have $q(\theta^*|\theta^{k-1})$ to be a ==symmetric proposal distribution== 
($q(x|y) = q(y|x)$).

>[!note] Metropolis
>- Start from some initial value $\theta^0$.
>- Repeat for $k=1$ to $K$:
>	1) Given the last value $\theta^{k-1}$, generate a **candidate sample** using the propositional distribution $$ \mathbf{\theta}^* \sim q(\mathbf{\theta}^*|\mathbf{\theta}^{k-1}) $$
>	2) Compute the **acceptance probability** $A_{k}$, $$ A_{k} = \text{min}\left( 1, \frac{p(\mathbf{\theta}^*)}{p(\mathbf{\theta}^{k-1})} \right) $$
>	3) Simulate $u_k \sim \mathcal{U}(0,1)$ and define $\theta^k$ as $$ \theta^k = \theta^*\text{ if }u_{k}<A_{k}\text{ otherwise defined as }\theta^{k-1} $$

The evaluation of $p(\theta)$ may be problematic if we are dealing with posterior distribution (because of the evidence term), but we can see it cancels out as the distribution is both in numerator and denominator.

**Questions left**: when to stop (convergence) and how many samples?