
# Diffusion-based Deep Generative Models (DDGMs)

Can briefly be explained as *hierarchical VAEs* bottom-up path defined by a ==diffusion process==, and a top-down path parameterized by DNNs, a ==reverse diffusion==.

The *bottom-up* path can be **fixed**, i.e., ==no learnable parameters==, for example adding simple Gaussian noise each layer. It circumvents the problem of *posterior collapse*, as the last layer should simply look like a *normal distribution of noise*!

## Model formulation

The general idea is to *iteratively destroy data*, and then learn the *reversing process* that ==restores== the data. As originally with Latent Variable Models, we are interested in finding the distribution over the data $p_\theta(\mathbf{x})$. Assume that we have $\mathbf{z}_{1:T}=[\mathbf{z}_1,\cdots,\mathbf{z}_T]$. Then the *marginal likelihood* is defined as:
$$ p_\theta(\mathbf{x}) = \int_{\mathbf{z}_{1:T}} p_\theta(\mathbf{x},\mathbf{z}_{1:T}). $$
The joint distribution inside the integral is modeled as a *first-order* Markov chain with *Gaussian* transitions:
$$ p_\theta(\mathbf{x},\mathbf{z}_{1:T})=p_\theta(\mathbf{x}|\mathbf{z}_1)\left( \prod_{i=1}^{T-1} p_\theta(\mathbf{z}_i|\mathbf{z}_{i+1}) \right) p_\theta(\mathbf{z}_T), $$
Where $\mathbf{x}\in\mathbb{R}^D$ and $\mathbf{z}_i\in\mathbb{R}^D$. Note, the latents are ***not bottlenecked in dimension***! All the distributions are parameterized with DNNs. This is a *hierarchical latent variable model*. 

The **variational posteriors** would look like:
$$ Q_\phi(\mathbf{z}_{1:T}|\mathbf{x}) = q_\phi(\mathbf{z}_1|\mathbf{x})\left(\prod_{i=2}^Tq_\phi(\mathbf{z}_i|\mathbf{z}_{i-1})\right) $$

However, the ==key point here== is *how* these distributions are defined. They are *not* defined by DNNs, but instead by Gaussian distributions, so ==no learnable parameters==:

$$ q_\phi(\mathbf{z}_i|\mathbf{z}_{i-1}) = \mathcal{N}\left(\mathbf{z}_i|\sqrt{1-\beta_i}\mathbf{z}_{i-1},\beta_i\mathbf{I}\right), $$
with $\mathbf{z}_0=\mathbf{x}$. The process is relatively simple; take the previous generated object, $\mathbf{z}_{i-1}$, and scale it $\sqrt{1-\beta_i}$ and then add noise variance $\beta_i\mathbf{I}$! With the *reparameterization trick*:
$$ \mathbf{z}_i=\sqrt{1-\beta_i}\mathbf{z}_{i-1}+\sqrt{\beta_i}\odot\epsilon, $$
with $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. The different $\beta$'s *could* be learnable parameters, but they can also be fixed. Papers suggest different, but one is for example linearly change from $\beta_1=10^{-4}$ to $\beta_T=0.02$. So $\beta_{t+1}>\beta_{i},\beta_{k}\in(0,1)$.

They are trained with **ELBO** as with VAEs!' The sum in the ELBO loss can be expensive, so using *law of unconscious statistician*, we replace it with a Monte Carlo estimation of a Uniform distributions:

$$ \mathbb{E}_{t\sim\mathbb{U}(2,T)}(T-1) = \sum_{t=2}^T\cdots$$
## Notes

Since we use Gaussian distributions and linear transformations, we can marginalize out intermediate steps (marginalize up to '$t-1$'):
$$ q(\mathbf{z}_t|\mathbf{x}) = \mathcal{N}\left( \mathbf{z}_t | \sqrt{\bar\alpha_t}\mathbf{x}, (1-\bar\alpha_t)\mathbf{I} \right), $$
Where $\alpha_t=1-\beta_t$, and $\bar\alpha_t = \prod_{s=1}^t\alpha_s = \prod_{s=1}^t(1-\beta_{t})$. This is smart, since we can forego the whole sampling process (which is iterative) and ==just sample directly==!

![[ComparisonOfModels.png]]
![[IndexVsStateSpace.png]]

# SDEs (ODEs)

==Not curriculum==
# Exercises

## 1)
From equation in the book, we have
$$ q(\mathbf{z}_{t}|\mathbf{x}) = \mathcal{N}(\mathbf{z}_{t}|\sqrt{ \bar{\alpha} }_{t}\mathbf{x},(1-\bar{\alpha}_{t})\mathbf{I}) $$
Where $\bar{\alpha}_{t}=\prod_{s=1}^t(1-\beta_{s})$. When the noise schedule is written as $0<\beta_{0}<\cdots<\beta_{T}<1$, then it falls to reason, as $T\rightarrow \infty$, that $\prod_{s=1}^t(1-\beta_{s})\rightarrow 0$.

Inserting this into the equation, it follows that:
$$ \begin{align}
q(\mathbf{z}_{t}|\mathbf{x}) &= \mathcal{N}(\mathbf{z}_{t}|\sqrt{0}\mathbf{x},(1-0)\mathbf{I})  \\
&= \mathcal{N}(\mathbf{z}_{t}|\mathbf{0},1\mathbf{I}) \\
&= \mathcal{N}(\mathbf{z}_{t}|\mathbf{0},\mathbf{I}) \\
\end{align} $$
## 2)
Derive equation (5) from equation (3) in the paper:

$$ \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t\geq_{1}}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right]  \tag{3}, $$
and 
$$ \mathbb{E}_{q}\left[ D_{KL}(q(\mathbf{x}_{T}|\mathbf{x}_{0})) + \sum_{t>1}D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})) - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1}) \right] \tag{5}. $$

If we start from (3), then
$$ \begin{align}
(3)&=\mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t\geq_{1}}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right] \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} -\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})}\right]  \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})}\cdot \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} -\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})}\right] \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})}+\sum_{t>1} \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} -\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})}\right]
\end{align} $$

If we now take a look at
$$ \begin{align}
-\sum_{t>1}\log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} &= -\sum_{t>1} (q(\mathbf{x}_{t-1}|\mathbf{x}_{0}) - q(\mathbf{x}_{t}|\mathbf{x}_{0})) \\
&= -(\log q(\mathbf{x}_{1}|\mathbf{x}_{0})-\log q(\mathbf{x}_{2}|\mathbf{x}_{0})) - (\log q(\mathbf{x}_{2}|\mathbf{x}_{0})-\log q(\mathbf{x}_{3}|\mathbf{x}_{0})) -\cdots-(\cdots-\log q(\mathbf{x}_{t}|))
\end{align} $$

What we see here is most terms cancel out except the very first, and the very last. If we now continue from where we left off, this means we have:
$$\begin{align}
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})}-\log q(\mathbf{x}_{1}|\mathbf{x}_{0}) +\log q(\mathbf{x}_{t}|\mathbf{x}_{0}) -\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})}\right] \\
&= \mathbb{E}_{q}\left[(\log q(\mathbf{x}_{t}|\mathbf{x}_{0})-\log p(\mathbf{x}_{T})) - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})} -\log q(\mathbf{x}_{1}|\mathbf{x}_{0}) - (\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1}) -\log q(\mathbf{x}_{1}|\mathbf{x}_{0}))\right] \\
&= \mathbb{E}_{q}\left[\log \frac{q(\mathbf{x}_{t}|\mathbf{x}_{0})}{p(\mathbf{x}_{t})}  - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})}  - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})\right] \\
&= \mathbb{E}_{q}\left[-\log \frac{p(\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}  - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})}  - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})\right] \\
&=  \mathbb{E}_{q}\left[ D_{KL}(q(\mathbf{x}_{T}|\mathbf{x}_{0})||p(\mathbf{x}_{T})) + \sum_{t>1}D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})) - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1}) \right]. 
\end{align}$$


## 3)