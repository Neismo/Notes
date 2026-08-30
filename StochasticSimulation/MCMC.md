
# Markov Chains
Components:
- $S$ state space (finite size $N$ or infinite)
- $T$ time index
- Transition Kernel $\mathbf{P}$.

The transition kernel is done such that:
$$ p_{ij} = \mathbb{P}(X_{t+1}=j|X_t = i),\qquad t\in T,\qquad i,j\in S. $$

Row signifies which state we are in, columns are probabilities of going to those states. When $\mathbb{P}$ does not change with time, it is *time-homogeneous*. This is good for simulation, as we can ignore time.

**Marginal distribution**: the marginal of the Markov chain at time step $n$ is
$$ \mathbf{P}(n) = \{ p_{i}(n) \}_{i\in S},\qquad p_{i}(n)=\mathbb{P}(X_{i}=n).$$
$$ \mathbf{p}(n) = \mathbf{p}(0)\mathbf{P}^n, $$
for some initial state distribution $\mathbf{p}(0)$.
**Stationary Distribution** a stationary distribution satisfies for some $\pi=\{ \pi_{i} \}_{i\in S}$
$$ \pi=\pi \mathbf{P} $$

**Limiting distribution**: at some point the marginal distributions becomes stationary after time

### Embedded Markov Chains (queue system)
For queue system:
- Arrival probability of next event: $\frac{\lambda}{\lambda+k\mu}$
- Departure probability of the next event: $\frac{k\mu}{\lambda+k\mu}$
As we can see, the probability of moving to state $i+1$ is the rate fraction of an arrival versus the combined of arrival and departure. 

## Global Balance Equations
For a single state we have for stationary distributions
$$ \pi_{j} = \sum_{i\in S}\pi_{i}p_{ij} $$
global balance equations are
$$ \sum_{i\in S}\pi_{i}p_{ij} = \sum_{i\in s}\pi_{j}p_{ji}, $$
implying that flow into state $j$ equals flow out of state $j$. A *local variant* exists also:
$$ \pi_{i}p_{ij} = \pi_{j}p_{ji}, $$
which reads that flow from $i$ to $j$ is equal to flow from $j$ to $i$. 
Local implies global (not other way). 
If local, the Markov chain is *reversible*.

# Markov Chain Monte Carlo
Markov Chain Monte Carlo allows us to generate samples from a tough $f(x)$, though not necessarily independently. From above, if we have a stationary MC, we can have a stationary distribution.

Given this distribution, we can construct the Markov chain that generates this stationary distribution. 

Covering *Metropolis-Hastings* (RW and Coordinate wise), and *Gibbs sampling*.

*General strategy*:
1) Construct Markov Chain with desired stationary distribution
2) Simulate the Markov Chain
3) Discard an initial *burn-in*
4) Use remaining states as samples from the target distribution 
	1) careful about sampling consecutive samples; dependent

*Idea*: we do not know $f(x)=cg(x)$, but we know $f(x)/f(y) = \frac{g(x)}{g(y)}$ i.e., the ratio, for all values $x,y$.
Using local balance, we need to find a $\mathbf{P}$ such that
$$ g(i)p_{ij} = g(j)p_{ji} $$

## Metropolis (Hastings) Algorithm
We want to sample from
$$ f(x)=cg(x) $$
1) *Propose* a candidate state $x$ from proposal distribution
2) Decide to accept/reject it
	1) Accept: move to $x$
	2) Reject: stay in $x$ and go again.

**Proposal distribution**: $h(x,y)=\mathbb{P}(x_{n+1}=y|X_{n}=x)$
- Should be easy to sample
- Easy to evaluate
- Explore the states efficiently
- Avoid excessive rejections

Commonly chosen such that it is symmetric: $h(x,y)=h(y,x)$

**Acceptance probability**
$$\alpha(x,y) = \min\left( 1, \frac{f(y)h(y,x)}{f(x)h(x,y)} \right) = \min\left( 1, \frac{g(y)h(y,x)}{g(x)h(x,y)} \right)$$
With a symmetric proposal distribution it simplifies:

$$\alpha(x,y) = \min\left( 1, \frac{g(y)}{g(x)} \right)$$

*Interpretation*:
- Moving to regions of higher probability are *always* accepted
- Moving to regions of lower probability are *sometimes* accepted

### Random Walk Metropolis Hastings
A simpler version is to generate a random perbutation from a symmetric distribution:
$$ Y_{i}=X_{i}+\Delta_{i} $$

since it is symmetric, the acceptance is
$$ \alpha(x,y) = \min\left( 1, \frac{g(y)}{g(x)} \right) $$

1) Generate proposal $Y_{i}= X_{i} + \Delta_{i}$
2) If $g(Y_i)\geq g(X_i)$, accept
3) Otherwise accept with probability $\alpha(x,y)$

# Gibbs Sampling
Suppose we wish to sample high-dimensional distribution
$$ f(x_{1},\dots,x_{d}) $$
Assume that conditional distributions are easy to calculate:
$$ \begin{align}
&f(x_{1}|x_{2},\dots,x_{d}) \\
&f(x_{2}|x_{1},x_{3},\dots ,x_{d}) \\
&\dots \\
&f(x_{d}|x_{1},\dots, x_{d-1})
\end{align} $$
Then the basic idea is to generate each coordinate at a time, based on the others.

*No rejection scheme*; we assumed we have correct conditional distributions.