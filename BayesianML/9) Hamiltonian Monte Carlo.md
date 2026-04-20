
# More on MCMC

As we saw, many summaries could be ==formulated as expectations== $\mathbb{E}_{p}[f(\mathbf{z})]$ for some function $f$. These expectation could be calculated with Monte Carlo Approximations

$$ \bar{f} = \mathbb{E}_{p}[f(\mathbf{z})] = \int f(\mathbf{z})p(\mathbf{z})\text{d}\mathbf{z} \approx \frac{1}{S}\sum_{i=1}^Sf(\mathbf{z}^i)\equiv \hat{f}, $$
where $z^i\sim p(\mathbf{z})$ for all $i$. It is ==unbiased== and variance of it ==decreases with number of samples==! Below is the ==Metropolis-Hastings Method==

>[!note] Metropolis-Hastings (M-H)
>- Start from some initial value $\theta^0$.
>- Repeat for $k=1$ to $K$:
>	1) Given the last value $\theta^{k-1}$, generate a **candidate sample** using the propositional distribution $$ \mathbf{\theta}^* \sim q(\mathbf{\theta}^*|\mathbf{\theta}^{k-1}) $$
>	2) Compute the **acceptance probability** $A_{k}$, $$ A_{k} = \text{min}\left( 1, \frac{p(\mathbf{\theta}^*)q(\theta^{k-1}|\theta^*)}{p(\mathbf{\theta}^{k-1})q(\theta^*|\theta^{k-1})} \right) $$
>	3) Simulate $u_k \sim \mathcal{U}(0,1)$ and define $\theta^k$ as $$ \theta^k = \theta^*\text{ if }u_{k}<A_{k}\text{ otherwise defined as }\theta^{k-1} $$
>Note that we now in this one have in the *acceptance probability* some new terms related to the propositional distribution. This is to allow *non-symmetrical* proposals, as if it is symmetrical it cancels out anyway.

From above it is clear that we need to be able to ==sample from the proposal distribution==, ==evaluate the proposal distribution==, and ==evaluate the target distribution==. 

For this we also require that the target distribution has the ==limiting distribution== attribute w.r.t Markov chains. This essentially means that we don't care where we initially are starting, we end up in the same place:
$$ p(\mathbf{z}^k) \to p(\mathbf{z})\text{ for } k\to \infty $$

The transition kernel for M-H is then (using $\mathbf{z}$ now):
$$ T(\mathbf{z'|z}) = \left\{\begin{matrix}
q(\mathbf{z'|z})A(\mathbf{z'|z}) & \text{if }z'\neq z \\
q(\mathbf{z|z})A(\mathbf{z|z})+\int q(\mathbf{z''|z})[1-A(\mathbf{z''|z})]\text{d}\mathbf{z}'' & \text{if } z'=z
\end{matrix}\right]$$

The first part is "if we propose a different state than the one we are in, the chance to transition to it is the chance of *proposing it* in the first place, and then the *chance of accepting it*"

The other part is "If we propose the same state, then it is composed of *EITHER* the chance of *proposing the same state* and then *accepting it* **PLUS** the chance of proposing *all other states* but not *accepting them*".

### Limiting Distribution & Ergodicity
For a Markov chain to have a limiting distribution we require.
1) It is **irreducible**: all states $\mathbf{z}$ can be reached
2) It is **Aperiodic**: no deterministic cycles
3) **Positive Recurrent**: Positive probability of returning to any given state (in finite time).
And then the Markov chain is **ergodic** and we have $p(\mathbf{z}^k)\to p^*(\mathbf{z})$. These are *hard to verify* in practice. 

The ==detailed balance condition== ensures *stationarity*:
$$ T(\mathbf{z'|z})p^*(z) = T(\mathbf{z|z'})p^*(z'), $$
which M-H can satisfy with proper proposition distributions.

So the problem is not *if we converge*, it is *when*.
# Advanced MCMC Sampling Algorithms
With M-H algorithm and a Gaussian proposal like so
$$ q(\theta^*|\theta^k) = \mathcal{N}(\theta^*|\theta^k,\tau^2\mathbf{I}) \text{ then } \theta^* = \theta^k + \tau \mathbf{\epsilon},\text{ } \mathbf{\epsilon}\sim\mathcal{N}(\mathbf{0,I}). \tag{1} $$

This is the ==Metropolized Random Walk==. It is simple and easy, but can suffer from *slow converging*. For low variance we may not move far and cover the distribution; for large variance we may drop many proposals.

Introducing ==Hamiltonian Monte Carlo (HMC)== (STATE OF THE ART?), a special case of the M-H algorithm. If you look above in equation (1) it is independent of *target distribution* $p(\theta)$. The HMC uses a smart *proposal algorithm* that *leverages the gradient of the target algorithm*!

More specifically, it utilizes the *potential energy*: $-\log p(\theta)$ and the idea is:
1) Choose point in distribution
2) Apply random momentum in a random direction
3) Follow the "dynamics" of the distribution for some steps
4) This is our new proposal (REPEAT).

Good demo showcase here: https://chi-feng.github.io/mcmc-demo/app.html. Next section we look at the *hows* of it!

# Hamiltonian Dynamics
>[!note] Required Definitions:
>- $p_{t}(\theta)$ be the target distribution for parameters of interest $\theta \in \mathbb{R}^D$. $\theta$ is a ==position== of some imaginative particle. 
> - $\nu$ be the ==momentum== of the particle, and it follows $p(\nu)=\mathcal{N}(\nu|\mathbf{0,I})$. 
> - $E(\theta)=-\log p_{t}(\theta)+\text{const}$ denote the ==potential energy== of the particle. 
> - $K(\theta)=-\log p(\nu)+\text{const}$ denote the ==kinetic energy==.

The ==Hamiltonian Function== is then the sum of energies, i.e., the ==total energy==:
$$\begin{align}
 H(\theta,\nu)&=E(\theta)+K(\theta)+\text{const}  \\
 &= -\log p_{t}(\theta)-\log p(\nu)+\text{const} \\
 \implies\exp^{-H(\theta,\nu)} &= \exp^{-\log p_{t}(\theta)-\log p(\nu)+\text{const}} \tag{Negate \& Exponent} \\
 &= p_{t}(\theta)p(\nu)\underbrace{\exp^{\text{const}}}_{\text{=Z}} \\
 \implies p_{t(\theta)}p(\nu) &= \frac{1}{Z}\exp^{-H(\theta,\nu)} \tag{Divide by Z}
\end{align}$$
Assume the *joint distribution* of the ==phase space== is given by the Hamiltonian because
$$p(\theta,\nu) = p_{t(\theta)}p(\nu) = \frac{1}{Z}\exp^{-H(\theta,\nu)} $$
### Hamilton's Equations
Describes how the particle with position $\theta$ and momentum $\nu$ moves over time:
$$ \begin{align}
\frac{\text{d}\theta}{\text{d}t} &= \frac{\partial H}{\partial \nu}=\frac{\partial K}{\partial \nu} \\
\frac{\text{d}\nu}{\text{d}t}&=-\frac{\partial H}{\partial \theta}=-\frac{\partial E}{\partial \theta}
\end{align} $$
So given initial state $(\theta_{0},\nu_{0})$ we can compute state $(\theta_{t},\nu_{t})$ for any given time $t$ by simulating/solving the above ODEs with ==discrete time== with ==Euler's Method==! It is possible to show that $\frac{\text{d}H}{\text{d}t} = 0$, meaning it is *conserved*, meaning $H_{0}=H_{t}!$

#### Euler's Method
Let $t\in\mathbb{R}$ denote continuous time index, and $k\in \mathbb{N}$ denote discrete time index. Let $\eta>0$ be the *step-size* and then define
$$ \frac{\text{d}\theta}{\text{d}t} \approx \frac{\theta(t+\eta) - \theta(t)}{\eta} \implies \theta(t+\eta) \approx \theta(t)+\eta\frac{\text{d}\theta}{\text{d}t} \implies \theta_{k+1}\approx \theta_{k}+\eta\frac{\text{d}\theta_{k}}{\text{d}t}$$

For *gaussian momentum* as we have, since $K(\nu) = \frac{1}{2}\nu^\top\nu+\text{const}$ we have that $\nabla_{\nu}K(\nu) = \nu$, and therefore
$$ \begin{align}
\theta_{k+1}&\approx \theta_{k}+\eta \nu_{k} \\
\nu_{k+1}&\approx \nu_{k}-\eta \nabla_{\theta}E(\theta_{k})
\end{align} $$
Due to *approximation error*, we *can't* turn it into a valid symmetric proposal algorithm unfortunately. Instead we try ==leapfrog method==

#### Leapfrog Method
Due to it being *reversible*, *volume preserving* and *having better approximation error*, we use this. Given a step size $\eta>0$ and a state $(\theta_{k},\nu_{k})$ at time $t=k \eta$, we can approximate the next state pair $(\theta_{k+1},\nu_{k+1})$ using a *leap-frog estimator*:
$$ \begin{align}
\nu_{k+\frac{1}{2}} &= \nu_{k} - \frac{\eta}{2}\nabla_{\theta}E(\theta_{k}) \tag{Half step for \(\nu\)} \\
\theta_{k+1} &= \theta_{k} + \eta \nu_{k+\frac{1}{2}} \tag{Full step for \(\theta\)} \\
\nu_{k+1} &= \nu_{k+\frac{1}{2}} - \frac{\eta}{2}\nabla_{\theta}E(\theta_{k+1}) \tag{Half step for \(\nu\)} \\
\end{align} $$
Above is ==1 step only== and we simulate/iterate for $T=L\eta$ steps for approximating state at time $T$.

This is a ==deterministic algorithm==, so we always end up at the same point. $\eta$ determines accuracy, lower is better, but requires more iterations. The number of steps $L$ and the step-size are *hyper-parameters*!

We wrote earlier that the Hamiltonian was *conserved*; this is only approximately conserving it (unlike Euler's) but that is fine in practice.
# Hamiltonian MC

How do we turn above into a *symmetric proposal distribution* $q(\theta^*,\nu^*|\theta_{k-1},\nu_{k-1})$? To *propose new candiate* we do
1) Sample fresh momentum $\nu\sim\mathcal{N}(\mathbf{0,I})$
2) Compute $(\theta^*,\nu^*)=\text{leapfrog}_{\eta,L}(\theta_{k-1},\nu_{k_{1-1}})$
3) Flip the momentum and return $(\theta^*,-\nu^*)$

We ==reverse momentum== due to ==reversibility==. Running leapfrog with from the new candidate will return the original one! Sometimes left out as Gaussians are symmetric and $p(\nu)=p(-\nu)$.

>[!note] Hamiltonian Monte Carlo (HMC)
> Let $\theta_{k}\in \mathbb{R}^D$ be the position at iteration $k$ and let $\text{leapfrog}_{L,\eta}(\cdot,\cdot)$ be a leapfrog integrator, then each iteration of HMC:
> For $k=1,2,\dots,K$ do:
> 1) Sample *new momentum* variable $\nu_{k-1}\sim\mathcal{N}(\mathbf{0,I})$.
> 2) Use *leapfrog integrator* to compute $(\theta^*,\nu^*)=\text{leapfrog}_{\eta,L}(\theta_{k-1},\nu_{k-1})$
> 3) Compute *acceptance probability* $a_{k}=\min(1,\exp[-H(\theta^*,\nu^*) + H(\theta_{k-1},\nu_{k-1})])$
> 4) Set $\theta_{k}=\theta^*$ with probability $a_{k}$ otherwise $\theta_{k}=\theta_{k-1}$.

The acceptance probability looks like that because recall that $p_{t}(\theta,\nu)=\frac{1}{Z}\exp(-H(\theta,\nu))$ and $a_{k} = \min\left(1,\frac{p_{t}(\theta^*,\nu^*)}{p_{t}(\theta_{k-1},\nu_{k-1})} \right) = \min\left(1,\frac{\frac{1}{Z}\exp(-H(\theta^*,\nu^*))}{\frac{1}{Z}\exp(-H(\theta_{k-1},\nu_{k-1}))} \right) = \min(1,\exp[-H(\theta^*,\nu^*) + H(\theta_{k-1},\nu_{k-1})])$
 
We also see that for small $\eta$ the acceptance probability will be close to 1!

There are a few hyper parameters to tune; $\eta$ and $L$. We can tune $\eta$ based on desired acceptance level $\delta$:
$$ \eta_{k+1} = \eta_{k} - \beta_{t}(\delta-a_{k}),\quad \beta_{t}=k^{-\gamma},\quad \gamma\in(0.5,1]. $$
==No U-turn Sampler NUTS== is a variant that tunes the L parameter by detecting when we are "turning" in the Hamiltonian Function space and then stops (in practice a little more involved)

Simpler is to randomize it or fix it, usually in $L\in[10,100]$.

# ULA & MALA
Two other algorithms

## Metropolis-Adjusted Langevin Algorithm (MALA)
Little brother of HMC, recall the leapfrog equations:
$$  \begin{align}
\nu_{k+\frac{1}{2}} &= \nu_{k} - \frac{\eta}{2}\nabla_{\theta}E(\theta_{k}) \tag{Half step for \(\nu\)} \\
\theta_{k+1} &= \theta_{k} + \eta \nu_{k+\frac{1}{2}} \tag{Full step for \(\theta\)} \\
\nu_{k+1} &= \nu_{k+\frac{1}{2}} - \frac{\eta}{2}\nabla_{\theta}E(\theta_{k+1}) \tag{Half step for \(\nu\)} \\
\end{align}  $$

What if $L=1$:
$$ \begin{align}
\theta_{k+1}^\text{MALA} &= \theta_{k} + \eta \nu_{k+\frac{1}{2}} \\
&= \theta_{k} + \eta \left(\nu_{k} - \frac{\eta}{2}\nabla_{\theta}E(\theta_{k})\right) \\
&= \theta_{k} - \frac{\eta^2}{2}\nabla_{\theta}E(\theta) + \eta \nu_{k} \\
&= \theta_{k} + \frac{\eta^2}{2}\nabla_{\theta}\log p_{t}(\theta) + \eta \nu_{k},\quad \nu\sim\mathcal{N}(\mathbf{0,I}).
\end{align}  $$

The ==Metropolis-Adjusted Langevin Algorithm (MALA)== is equivalent to:
$$ q_{\text{MALA}}(\theta^*|\theta_{k}) = \mathcal{N}\left( \theta^*|\theta_{k}+\frac{\eta^2}{2}\nabla_{\theta}\log p_{t}(\theta_{k}),\eta^2\mathbf{I} \right), $$
so the distribution has a *bias* towards the gradient of the target distribution, pushing towards high-density regions. In regions of low density, the gradient may dominator over the variance and push towards high-density regions.

It is still not too scaleable, as we require a full sweep through the dataset for acceptance probability.
## Unadjusted Langevin Algorithm (ULA)

Same as MALA, but skips the metropolis acceptance/rejection step! This mean we can mini-batch gradients, but no longer any guarantee. Faster but biased and may not converge. Smaller step-size can help.

![[Comparison Of Algorithms for MCMC.png]]

# Convergence Diagnostics