
# Discrete
**Direct crude** method:
- Can take values $x_1,\cdots,x_k$ with probabilities $p_1,\cdots,p_k$
- When U (random number) falls into interval of length $p_i$
- $$X=x_i,\text{ if } F(x_{i-1})<U<F(x_i)$$
- $$X=x_i,\text{ if } \sum_{j=1}^{i-1}p_j<U<\sum_{i}^{j=1}p_j$$
**Rejection** method:
- Again $k$ classes with $k$ probabilities $p_1,\cdots,p_k$.
- Need number $c > \max_i{p_i}$ that is larger than all probabilities for classes.
- Generate $I = floor(kU_1)+1$
	- If $U_2 \leq p_I/c$, then $X=I$ else go again.
- Essentially we select class $k$ equally, then compare uniform to C. Problem if $c$ is very large?
- **General** rejection:
	- Change proposal distribution Q (no longer choose uniformly among classes)
	- Choose $c > \max_i\{p_i/q_i\}$
	- Generate according to proposal: $I = Y$
	- If $U < p_I/cq_I$, then $X=I$ else go again.
**Alias** method:
- Pick proposal (among classes)
- Uniform number to some place; if below, accept, if above, pick new number/class ()
# Continuous
**Inverse** transformation Technique
- *Idea*: do inverse of CDF; first sample U(0,1) then find the associated value from the CDF.
- *Require*: it has to be invertible.
- $Y = F_X^{-1}(U)$, where $Y=X$, so we can generate from distribution of $X$ with uniform $U$ through its inverse!
- For exercise: need the mean and variance of *pareto*.
- For Rayleigh: samples **S** $\sim$ exp(1/2), then sqrt to get **R**

**Mathematical** methods:
- **Box-Mueller**: transform two uniforms to two independent normal univariates
- **Central Limit Theorem**: add enough uniform variables, standardize with mean/var, then you have a standard normal variable (go with high $n$).

We now with ways to sample Gaussian $Z$:
- **Log-normal**: $X = \alpha + \beta Z$, and let $Y=e^X$. X is normal
	- $\log(Y)\sim N(\alpha,\beta^2)$, $Y\sim LN(\alpha,\beta^2)$.
- **Multivariate**: Use Cholesky factorization we can use a multivariate normal from 

**Composition** methods:
- If the desired distribution function is a *composition*/*mixture* of simpler ones;
	- first $U_1$ sample which of them we use with probability $p_i$. so $I\sim\{p_i\}_i^k$
	- Then with $U_2$ sample from *that* distribution 
	- *take-away*: if its easy to do conditional and marginals, then composition is nice!
- **Hyper exponential**: different rate parameters
- **Erlang**: same rate parameter

**Rejection** methods:
- Requires a simpler sampling density $g(x)$ than $f(x)$.
- *Assumption*: $f(y)/g(y)<c$ for all $y$.
- Algorithm:
	- Generate $Y$ with density $g$ (so $Y\sim g$)
	- Generate $U\sim U(0,1)$
	- If $U \leq f(Y)/cg(Y)$ then accept, otherwise repeat
		- if accept: $X=Y$
- *Intuition*: we have proposal, true and the scaled one by C
	- Sample from proposal.
	- Draw uniform from axis up to *scaled* proposal.
	- If U falls below the true one, accept, otherwise reject.