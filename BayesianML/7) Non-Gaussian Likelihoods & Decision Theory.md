
# Gaussian Processes & Neural Networks

Neural Networks and GPs are similar near regions of data density, but outside they are not; neural networks are overconfident, but scale well; GPs take the uncertainty into account, but scale poorly.

Take a NN with a single hidden layer with $H$ neurons and a single output is formalized as
$$ \begin{align}
\mathbf{\mathbf{z}}(\mathbf{x})=h(\mathbf{W}_{1}\mathbf{x} + \mathbf{b}_{1}) \\
f(\mathbf{x}) = \mathbf{W_{2}}\mathbf{z}(\mathbf{x})+\mathbf{b}_{2},
\end{align} $$
for $\mathbf{x}\in \mathbb{R}^D,\mathbf{W}_{1}\in \mathbb{R}^{H\times D},\mathbf{W}_{2}\in \mathbb{R}^{H\times_{1}}$. If $H$ is large enough, then this is considered a ==universal approximater== for many classes of functions. Let's make some assumptions
1. Activation function $h$ is bounded - $h(x)=\tanh(x)$
2. $\mathbf{W}_{1},\mathbf{b}_{1}$ have i.i.d zero-mean priors
3. $\mathbf{W}_{2},\mathbf{b}_{2}$ have zero-mean and prior variances $\sigma^2_{\mathbf{w}},\sigma^2_{\mathbf{b}}$
4. Prior variance of $\sigma^2_{\mathbf{w}}=\frac{1}{H}$

Then for example mean is
$$ \mathbb{E}[f(\mathbf{x})] = \mathbb{E}[\mathbf{W_{2}}\mathbf{z}(\mathbf{x})+\mathbf{b}_{2}] = \sum_{i=1}^H \underbrace{\mathbb{E}[w_{i}]}_{0}\cdot\mathbb{E}[h_{i}(\mathbf{x})]+\underbrace{\mathbb{E}[\mathbf{b}_{2}]}_{0} = 0, $$
and is so a zero-mean stochastic process. As $H\to \infty$, the neural network converges to a Gaussian process due to Central Limit Theorem with covariance function
$$ k_{\text{NN}}(\mathbf{x},\mathbf{x}') = \frac{2}{\pi}\sin^{-1}\frac{2\tilde{\mathbf{x}}^\top\tilde{\mathbf{x}}'}{\sqrt{ (1+2\tilde{\mathbf{x}}^\top \tilde{\mathbf{x}})\cdot(1+2\tilde{\mathbf{x}}'^\top \tilde{\mathbf{x}}') }},\quad \tilde{\mathbf{x}} = \begin{bmatrix}
1,\mathbf{x}
\end{bmatrix}^\top, $$
comparing the squared exponential (left) this looks like the following (NN on right):
![[NNvsGP.png]]

# Generalized Linear models & Non-Gaussian Likelihoods

==Generalized Linear Model== (GLM) when appropriate for ones data; regression with Gaussian assumptions is not always, for example (prone to outliers).
$$ y_{n}|f_{n} \sim p(y_{n}|f_{n}) $$

Below are some *common likelihoods* in ML
![[CommonLikelihoods.png]]

Components of a *GLM*:
1) The linear model itself
$$ f(\mathbf{x}) = \phi(\mathbf{x})^\top\mathbf{w} $$
2) The ==link function== $g$ that relates the **mean of the linear model** with the **mean of the response variable** $\mathbb{E}[y(\mathbf{x})|\mathbf{x}] = \mu(\mathbf{x})$
$$ g(\mu(\mathbf{x})) = f(\mathbf{x})\quad\Leftrightarrow\quad \mathbb{E}[y|\mathbf{x}] = g^{-1}(f[\mathbf{x}]) $$
3) The distribution of $p(y_{n}|\mathbf{x}_{n})$ for the response variable i.e., Poisson, Gamma, Gaussian etc.

If the mean $\mu=0$, then $g$ is the identity function, if $\mu>0$ then choose $g$ to be the log link function ($g^{-1}=\exp$), and if $0<\mu<1$ in the unit interval, then $g$ is the sigmoid function or standard Gaussian CDF.

Step 1 we can also replace for Gaussian Process with $f(\mathbf{x})\sim\mathcal{GP}(0,k(\mathbf{x},\mathbf{x}'))$. We can adapt the GP to different likelihoods easily! From last week, we did Gaussian Process on classification, where we had
$$ p(\mathbf{f}|\mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{f})p(\mathbf{f})}{p(\mathbf{y})}\approx q(\mathbf{f}) = \mathcal{N}(\mathbf{f}|\mathbf{m},\mathbf{S}), \tag{Laplace Approx} $$
and from that, the log joint of the target $\mathbf{y}$ and the latent function values $\mathbf{f}$
$$ \log p(\mathbf{y,f}) =\log p(\mathbf{y|f})+\log p(\mathbf{f}) = \sum_{n=1}^N\log p(y_{n}|f_{n}) - \frac{N}{2}\log p(2\pi) - \frac{1}{2}|\mathbf{K}|-\frac{1}{2}\mathbf{f}^\top\mathbf{K}^{-1}\mathbf{f}. $$
We needed the laplace, so we computed the gradient and Hessian for the above and get
$$\begin{align}
\nabla_{\mathbf{f}}\log p(\mathbf{y,f}) &= \sum_{n=1}^N\nabla_{f}\log p(y_{n}|f_{n})-\mathbf{K}^{-1}\mathbf{f} \\
\nabla^2_{\mathbf{f}}\log p(\mathbf{y,f}) &= \sum_{n=1}^N\nabla_{f}^2\log p(y_{n}|f_{n})-\mathbf{K}^{-1},
\end{align}$$
so for different likelihoods, we just need to be able to evaluate those two terms, and we can reuse the computations otherwise! So the first and second order derivative of the log likelihood.
# Generalization & Evaluation

In supervised learning, consider the dataset $\mathcal{D}=\{\mathbf{x}_{i},y_{i}\}^N_{i=1}$

We measure performance using *cross validation*
- Held-out test set
- K-fold
- Leave-one-out
- Split-half
- And more

**Goal**: assessing the ==generalization error== of the new model; how well does it perform on unseen data. For **parameter tuning**: test, validation and training sets or *nested cross-validation*.

## Generalization Error & Loss Functions
Let $y$ be the target value for a input vector $\mathbf{x}$ and let $\hat{y}\equiv \hat{y}(\mathbf{x})$ be a prediction. The ==loss function== $\mathcal{L}(y,\hat{y})$ defines the cost of predicting $\hat{y}$ when the true value is $y$. Different losses could be
1) The *quadratic loss* for regression which is
$$ \mathcal{L}(y,\hat{y}) = (y-\hat{y})^2 $$
2) The *0/1 loss* for classification which is
$$ \mathcal{L}(y,\hat{y}) = \mathbb{I}[y\neq \hat{y}] $$

The ==generalization error== (or **expected loss**, **risk**, **out-of-sample error**) for a model $\hat{y}(\mathbf{x})$ is defined 
$$ \mathcal{R}_{\hat{y}} \equiv \mathbb{E}[\mathcal{L}(y,\hat{y}(\mathbf{x}))] = \iint \mathcal{L}(y,\hat{y}(\mathbf{x}))p(\mathbf{x},y)\text{d}\mathbf{x}\text{d}y, $$
where we rarely know the distribution of $p(\mathbf{x},y)$.

In practice, we can estimate it. If we assume i.i.d test samples $(\mathbf{x}_{i}^*,y_{i}^*)\sim p(\mathbf{x},y)$, $i=1,\dots,N_{\text{test}}$ then
$$\mathcal{R}_{\hat{y}}\approx \hat{\mathcal{R}}_{\hat{y}}^{\text{test}} = \frac{1}{N}\sum_{i=1}^{N_{\text{test}}}\mathcal{L}(y_{i}^*,\hat{y}(\mathbf{x}_{i}^*)),$$
which is an accurate estimator for $N_{\text{test}}\to \infty$! Similar for ==empirical risk of training data==
$$ \mathcal{R}_{\hat{y}}\approx \hat{\mathcal{R}}_{\hat{y}}^{\text{train}} = \frac{1}{N}\sum_{i=1}^{N_{\text{train}}}\mathcal{L}(y_{i},\hat{y}(\mathbf{x}_{i})). $$

Much learning uses this as framework for optimizing known as ==empirical risk minimization (ERM)==
$$ \hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \hat{\mathcal{R}}_{\hat{y}}^{\text{train}}. $$
In ERM, the risk is *optimisitc* $\mathbb{E}[\hat{\mathcal{R}}_{\hat{y}}^{\text{train}}]\leq\mathbb{E}[\hat{\mathcal{R}}_{\hat{y}}^{\text{test}}]$.

### Example with Regression loss
What would the generalization error be with regards to *regression loss*? Assume the *simple* linear model of 
$$ y=w_{\text{true}} + \epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma^2) $$

We will use an estimator $\hat{w}$ for $w_{\text{true}}$ (fixed) as a prediction for new data $y^*=\hat{w}$.

$$ \begin{align}
\mathcal{R}_{\hat{w}} = \mathbb{E}[\mathcal{L}] &= \iint (y-\hat{w})^2p(\mathbf{x},y)\text{d}y\text{d}x \\
&=\underbrace{\int(y-\hat{w})^2p(y)}_{\text{marginalized }\mathbf{x}\text{ out}}  \tag{2} \\
&=\int(y-\hat{w})^2\mathcal{N}(y|w_{\text{true}},\sigma^2)\text{d}y \\
&= \int(y^2-2y\hat{w}+\hat{w}^2)\mathcal{N}(y|w_{\text{true}},\sigma^2)\text{d}y \\
&= \underbrace{\int y^2\mathcal{N}(y|w_{\text{true}},\sigma^2)\text{d}y}_{\text{Second momentum}} + \hat{w}^2\underbrace{\int\mathcal{N}(y|w_{\text{true}},\sigma^2)\text{d}y}_{\text{Dist must sum to 1}} - 2\hat{w}\underbrace{\int y\mathcal{N}(y|w_{\text{true}},\sigma^2)}_{\text{Definition of Mean}} \\
&= (w_{\text{true}} - \hat{w})^2+\sigma^2
\end{align} $$
which is unsurprisingly minimized when $\hat{w}=w_{\text{true}}$, and so $\mathbb{E}[\mathcal{L}]=\sigma^2$.

# Decision Theory
## Uncertainty in multi-class classification
Categorical distribution for K-classes in multi-class classification
$$ y_{n}|\mathbf{f}_{n} \sim \text{Categorical}[\text{Softmax}(\mathbf{f}_{n})] $$

The posterior predictive function for this is another categorical with
$$ p(y^*=k|\mathbf{y},\mathbf{x}^*) = \pi_{k},\quad\text{ for }\quad \sum_{k=1}^K\pi_{k}=1\text{ and }0\leq \pi_{k}\leq_{1} $$

Predicting most likely class can be a simple arg max
$$ \hat{y}^* = \arg\max_{k}P(y^*=k|\mathbf{y},\mathbf{x}^*) $$Depending on the task, we might want to be careful with this (healthy vs cancer for example). Simpler quantities may help with this by accounting for uncertainty

>[!note] Confidence of a Posterior Predictive Distribtution
>$$ \mathcal{C} = \max_{k} p(y^*=k|\mathbf{y},\mathbf{x}^*) $$
>This ranges from $\left[ \frac{1}{K},1 \right]$. This is because the *max* operation is lowest in a uniform distribution.

>[!note] Entropy
>$$ \mathcal{H} = -\sum_{k=1}^K\pi_{k}\log\pi_{k}, $$
>where *higher* is *more uncertain*. Ranges from $[0, \log K]$.
>Convention when $\pi_{k}=0:0\cdot \log0=0$.

>[!faq]- Example of *confidence* and *Entropy*
>![[ConfidenceEntropyExample.png]]
>Despite the same *confidence* in their maximum predictions, entropy is higher in the right case, because there is more uncertainty around the rest of the predictions.

Sometimes there is risk in making a decision and we have the option of a ==reject opinion==: avoiding a decision if the uncertainty is too large. Simply it could be to reject if *confidence* is low i.e.,
$$ \mathcal{C} < \theta_{\text{reject}}, $$
where if $\theta_{\text{reject}} = 1$ then all samples are rejected, and if $\theta_{\text{reject}}\leq\frac{1}{K}$ then no samples are rejected.

## Decision Theory Formally
In Bayesian Modelling we strive to represent all unknown quantities (parameters, predictions etc.) using *probability distributions*; often we must make binary decisions/predictions however.

==Statistical Decision Theory== tells us to make optimal decisions under uncertainty

>[!note] Decision Theory Framework
>A ==decision maker== has a set of **actions/choices**  $a\in\mathcal{A}$ to choose from.
>The optimal decision depends on the **true state** $s\in\mathcal{S}$ of the system.
>The **loss function** $\mathcal{L}(s,a)$ determines the **cost** for a *choice a* when the *true state* is s. It can also be formulated with a **utility function** $\mathcal{U}(s,a)=-\mathcal{L}(s,a)$.
>
>Example below for COVID-19
>![[DecisionTheoryExample.png]]
>
>In practice we do not know the true state *s*.
>
>>[!note] Bayesian Decision Theory
>>1) Compute posterior of the state *s* given the data $\mathbf{y}$, $p(s|\mathbf{y})$
>>2) Choose action that minimizes the posterior expected loss 
>>$$\hat{a}=\arg\min_{a\in\mathcal{A}}\mathbb{E}_{p(s|\mathbf{y})}[\mathcal{L}(s,a)]$$


### Binary Classification
Consider classification so $y_{i}\in\{0,1\}$. 
The ==0/1 utility function== is given by $\mathcal{U}(y,\hat{y}(\mathbf{x})) = \mathbb{I}[y=\hat{y}(\mathbf{x})]$. If we denote $p\equiv p(y^*=1|\mathbf{y},\mathbf{x}^*)$ - the probability of *true* label 1 - for the posterior class probability for input point $\mathbf{x}^*$, then
$$ \mathbb{E}_{p}[\mathcal{U}(y^*,\hat{y}(\mathbf{x}))] = \sum_{y^*}p(y^*|\mathbf{y},\mathbf{x}^*)\mathcal{U}(y^*,\hat{y}(\mathbf{x})) = (1-p)\mathbb{I}[0=\hat{y}(\mathbf{x})]+p\mathbb{I}[1=\hat{y}(\mathbf{x})] $$

This means the expected utility is for our two prediction classes:
$$ \begin{align}
\mathbb{E}_{p}[\mathcal{U}(y^*,1)] &= p \\
\mathbb{E}_{p}[\mathcal{U}(y^*,0)] &= 1-p
\end{align} $$
Which means picking highest predictive probability is *Bayes' optimal* under the 0/1 utility function.

Assume instead we are in the medicinal industry classifying cancer, where being wrong is bad; consider then this new utility function, still in *binary classification*
![[UtilityFunctionExample.png]]

Then we can ask "how certain before I dare predict not cancer"?
$$ \begin{align}
\mathbb{E}[\mathcal{U}(y^*,0)]&\geq\mathbb{E}[\mathcal{U}(y^*,1)] \\
\Rightarrow (1-p)\mathcal{U}_{00}+p\mathcal{U}_{10}&\geq(1-p)\mathcal{U}_{01}+p\mathcal{U}_{11} \\
\Rightarrow p&\leq\frac{\mathcal{U}_{01}-\mathcal{U}_{00}}{\mathcal{U}_{01}-\mathcal{U}_{11}+\mathcal{U}_{01}-\mathcal{U}_{00}}\approx0.099, 
\end{align} $$
which means that if $1-p>0.901$ then we can predict "no cancer". This is far greater than $0.5$ we saw before which is the usual scenario.

### Regression
Consider now a regression problem so $y_{i}\in \mathbb{R}$. The most common loss function is the *quadratic loss* function
$$ \mathcal{L}(y,\hat{y}(\mathbf{x})) = (y-\hat{y}(\mathbf{x}))^2, $$
and the expected loss is then
$$ \mathbb{E}[\mathcal{L}(y,\hat{y}(\mathbf{x}))] = \iint (y-\hat{y}(\mathbf{x}))^2p(y,\mathbf{x})\text{d}x\text{d}y, $$
What is the optimal predictor function? It can be shown to be the *mean*
$$ \hat{y}(\mathbf{x}) = \mathbb{E}[y|\mathbf{x}] $$

The loss can be *generalized* as the ==Minkowski== loss:
$$ \mathcal{L}(y,\hat{y}(\mathbf{x})) = (\hat{y}(\mathbf{x})-y)^q, $$
For $q=2$, the optimal is at the *posterior mean*
For $q=1$, the optimal is at the *posterior median*
For $q=0$, the optimal is at the *posterior mode*
- For $q=2$, it is sensitive to outliers, where as $q=1$ is more robust to them.
# Calibration

For classification models. Trained models are in practice not well calibrated. We would expect that if the predictive probability is around 80% for all examples in the test set, then we expect roughly 80% of the corresponding examples to belong to the positive class.

Metrics for quantifying miscalibration for classification:
- Expected Calibration Error (ECE)
- Maximum Calibration Error
- Marginal Calibration Error
- Brier Score

## Expected Calibration Error (ECE)
Compute predictions for the validation set $\mathcal{D}_{\text{val}}=\{\mathbf{x}_{m}^*,y_{m}^*\}_{m=1}^M$. This gives a density over the probabilities in the unit space. 
Next step is to divide that unit interval into $B$ bins such that $I_{b}=\left(\frac{\frac{b-1}{B},b}{B}\right]$.
Let $\mathcal{B}_{b}$ be the set of indices of samples whose prediction confidence fall into the interval and define
$$ \begin{align}
\hat{y}_{m}^* &= \arg\max_{c}p(y_{m}^*=c|\mathbf{y},\mathbf{x}^*_{m}) \\
\mathcal{C}=\hat{p}_{m}^* &= \max_{c} p(y^*_{m}=c|\mathbf{y},\mathbf{x}_{m}^*) ,\tag{Confidence}
\end{align} $$
So the predictions and its confidences.
Then ==average accuracy== for bin $b$ is defined as
$$ \text{acc}(\mathcal{B}_{b}) = \frac{1}{|\mathcal{B}_{b}|}\sum_{m\in\mathcal{B}_{b}}\mathbb{I}[\hat{y}^*_{m}=y^*_{m}], $$
as well as the ==average confidence== for bin $b$:
$$ \text{conf}(\mathcal{B}_{b})=\frac{1}{|\mathcal{B}_{b}|}\sum_{m\in\mathcal{B}_{b}}\hat{p}_{m}^*$$
Where the ==expected calibration error ECE== is given by
$$ \text{ECE} = \sum_{b=1}^B \frac{|\mathcal{B}_{b}|}{M}|\text{acc}(\mathcal{B}_{b}) - \text{conf}(\mathcal{B}_{b})| $$
## Recalibration
Several methods
- Temperature Scaling
- Platt Scaling
- Histogram Binning
- Isotonic Regression

In ==temperature scaling==, we introduce a temperature paremeter $\tau>0$ in the softmax function
$$ p(y|f(\mathbf{x}),\tau) = \text{Categorical}\left( y|\text{Softmax}\left( \frac{f(\mathbf{x})}{\tau} \right) \right), $$
where $\tau$ is estimated on the validation data set.


# Summary
![[Summary7.png]]

# Lecture Exercise
![[lectureExercise.png]]
## What is the posterior expected loss for each action i.e. $\mathbb{E}_{p(s|\mathbf{y})}[\mathcal{L}(s,a)]$
We calculate it directly so for
$$\begin{align}
\text{Do Nothing}&: 0.65\cdot 0+0.3\cdot 10+0.05\cdot 100 = 8 \\
\text{Isolate}&: 0.65\cdot 10+0.3\cdot 0+0.05\cdot 10 = 7 \\
\text{Hospitalize}&: 0.65\cdot 50+0.3\cdot 20 + 0.05\cdot 0=38.5
\end{align}$$

## What is the optimal action? $(\hat{a}=\arg\min_{a\in\mathcal{A}}\mathbb{E}_{p(s|\mathbf{y})}[\mathcal{L}(s,a)])$
The best action is to *isolate* at home!