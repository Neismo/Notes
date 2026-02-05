
**Keywords**: Bias, Variance, Bias-Variance Trade-off, Expected Prediction Error (EPE), Linear Regression, OLS, Ridge Regression, M vs N.
# Goals & Terminology

**The goal**: learning from data $\mathcal{D} = \{(x_i,y_i)\}_{i=1}^n$.
**The split**: learning $\leftarrow$ Training vs Test

## **Terminology**:
- Based on nature of the ==*response variable*==:
	- **Unsupervised**: no response variable *available*
		- PCA
		- Cluster analysis
	- **Supervised**: Response variable(s) *available*
		- Categorical response (**classification**):
			- Linear Discriminant Analysis
			- K Nearest Neighbors
			- Support Vector Machines
		- Continuous response (**Regression**):
			- Ordinary Least Squares
			- Ridge Regression
			- K-nearest-neighbors

### **Error types**:
- **Training errors**: optimistic estimate
- **Test errors**: generalization performance
- **Generalization gap**: difference between the two.

### Data notation
- **Matrix**: $X$ (capital letter, non bold), $n\times m$
	- Row is $n$ **samples**, columns $m$ **variables**
- **Response variable**: $y$ (small letters denote *vectors*)
	- Usually a $(n\times 1)$ vector, also called **output**
- **Model coefficients**: $\beta$
	- $(m\times 1)$ vector
- **Model error**: $e$ 
	- $(n\times 1)$ vector
- **Prediction error**: $\epsilon$
	- $(n\times 1)$ vector, also known as **residual**
- **Regression model** linear in $\beta$:
	- $y=X\beta + e$

## Instability under Resampling

**Observation**: small changes in $\mathcal{D}$ lead to large changes in $\hat{f}$. *Training error* is too optimistic, while *test error* is only for a snapshot of the dataset. Multiple fitted curves show *variation* across different samples.

### Expected Prediction Error (EPE)

Theoretical ==gold standard== of target:
$$ EPE(\hat{f}) = \mathbb{E}\left[Y - \hat{f}(X)^2\right] $$
- **Problem**: we can't integrate over $P(X,Y)$. We only have *n* samples.
- **Goal**: model that minimizes EPE!

#### Lower Bound: irreducible noise
The true process *usually* includes some ==noise $\epsilon$ which we can't reduce==. So $EPE>0$. The variation in this noise is $Var(\epsilon) = \sigma^2$.
$$\text{Total Error} = \text{Reducible Error} + \sigma^2$$
#### Decomposing EPE:
Three source of errors contribute:
$$\begin{align} EPE(x_0) &= E_{y,\mathcal{D}|x_0}|| (y(x_0) - \hat{f}(x_0;\mathcal{D}))^2 || \\
&= \sigma_e^2+\text{Bias}^2(\hat{f}(x_0;\mathcal{D}))+\text{Variance}(\hat f(x_0;\mathcal{D}))
\end{align}$$
1) **Irreducible error**: $\sigma_e^2=E_{y}(y(x_0) - f(x_0))^2$
2) **Bias:** $\text{Bias}^2(\hat{f}(x_0;\mathcal{D})) = \left( E_\mathcal{D}(\hat f(x_0;\mathcal{D})) - f(x_0)\right)^2$
3) **Variance**: $\text{Variance}(\hat f(x_0;\mathcal{D})) = E_\mathcal{D}(\hat f(x_0;\mathcal{D}) - E_\mathcal{D}(\hat f(x_0;\mathcal{D})))^2$,

Where $\hat f(x_0)$ is the estimate of $f(x_0)$ with observed $y_0=f(x_0)+e, E(e)=0$, variance(e) = $\sigma_e^2$ and $\mathcal{D}$ is the training data.

# Statistical Bias
Difference between an expected value and the *true* value:
$$ \text{Bias}(\hat\theta) = \mathbb{E}[\hat\theta]-\theta $$

to get an *unbiased* estimate:
- Repeat experiments, take average of $\theta$.

## Variance
Quantifies how far we are from target - we may be unbiased on average, but can be quantified.
$$ Var(\hat\theta) = \mathbb{E}\left[(\hat\theta - \mathbb{E}[\hat\theta])^2\right] $$

# Linear Regression
Given a continuous response variable, find relations to a set of input variables:
$$ y=X\beta + e $$
### **Ordinary Least Squares** (OLS)
finding a $\beta$ that minimizes the *residual error* $y - X\beta$. Minimize:
$$ ||y-X\beta||^2_2 = \sum_{i=i}^n\left( y_i - X_i\beta \right)^2,$$ from which we can determine by setting gradient to 0, and isolating $\beta$:
$$ \beta_{OLS} = \arg\min_\beta||y-X\beta||_2^2$$
$$\begin{align}
\frac{\text{d}}{\text{d}\beta}(y-X\beta)^\top(y-X\beta) &= 0\\
-2X^\top(y-X\beta) &= 0 \\
X^\top y-X^\top X\beta &= 0 \\
X^\top y &= X^\top X\beta \\
\beta_{ols}&=(X^\top X)^{-1}X^\top y
\end{align}$$

**Requirement**: $X^\top X$ *must* be invertible
**Instability**: if features are highly correlated, then $X^\top X$ may be near singular and have high; causes *high variance*.

**OLS** is *great*: ==best linear unbiased== estimate!

## Controlling bias-variance

###  Model Complexity
The *flexibility* or *degrees of freedom* of the estimator $\hat f$.
- **Parametric models**: often tied to number of parameters *m*
- **Non-parametric models**:  For KNN, as example, it is tied to *number of neighbors considered*

**Trade-off**:
- **Low complexity**: High bias, low variance $\rightarrow$ underfitting
- **High complexity**: Low bias, high variance $\rightarrow$ overfitting

#### M vs N Learning Moment

The complexity is not an absolute property, but related to the amount of data *n*:
**Stable Regime** $(n \gg m)$:
- Data constrains the model
- Training and test errors are close
- Overfitting is *difficult*

**Overfitting Regime** $(n\approx m)$:
- Model fits *noise*
- *Large* generalization gap
- Instability explodes

## Ridge Regression
Ridge regression adds a term, $\lambda$, to $\beta$ helping lowering its size and by that the *variance*. The solution to ridge regression is:

$$ \beta_{\text{ridge}} = (X^\top X+\underbrace{\lambda I}_{\text{Diag}})^{-1} X^\top y $$

Intuitively, we add a small value to the diagonal of the matrix we invert.
- Stabilizes the inverse numerically, and is the reason it is called *ridge* regression.
- Ridge regression solutions are available even when $m > n$.

**Important**: center & normalize data!


#### Derivation of Ridge Regression Parameter (Exercise):
We set the derivation of the ridge regression expression with respect to $\beta$ and set it to 0. Then we isolate for $\beta$:
$$\begin{align}
\frac{\text{d}}{\text{d}\beta}(y-X\beta)^\top(y-X\beta) +\lambda\beta^\top\beta &= 0\\
-2X^\top(y-X\beta)+2\lambda\beta &= 0 \\
-X^\top(y-X\beta)+\lambda\beta &= 0 \\
-X^\top y+X^\top X\beta + \lambda\beta &= 0 \\
-X^\top y+(X^\top X+\lambda I)\beta &= 0 \\
(X^\top X+\lambda I)\beta &= X^\top y \\
\beta_{\text{ridge}} &= (X^\top X+\lambda I)^{-1}X^\top y
\end{align}$$

# Model Selection

What is the *optimal* level of complexity? We can't observe EPE directly!
**Solution**: using ==validation sets== and ==cross validation== to estimate test performance / EPE


# Exercises

## 1) OLS and the ’Wobble’ of Instability
- **Instability**: If there is *no* irreducible error introduced via $\sigma$, then the solutions have no variance, but the variance scales with the variance of the error. When it is very high, of course there is so too in the model coefficients.
- **Identifiability**: Changing the value of $\rho$ from being 0 to close to 1, introducing co-linearity between the two features introduced a lot of variance in the coefficients of the model; when it was 0, the variance in its coefficient were low, and centered around the true solution.
- **Unbiased vs Error**: even if we have an unbiased estimator like OLS, we can still have a high prediction error due to the variance term. If we can reduce this, by adding a bit of bias, then we might get a lower prediction error.

## 2)  Ridge Regression and Controlled Bias
- **Bias-Variance Trade-off**: as $\lambda\rightarrow\infty$, we see that both the squared bias and the variance tends towards a low value. 
- **Regularization Path**: bias² seems to shrink faster than variance.


## 3) The $M$ vs $N$ (Wine Quality)
- **High data - low features**: here OLS compares decently with Ridge Regression. Both achieve low train/test error.
- **Many features to small amount of data**: here OLS breaks down completely, achieving very good train MSE, but very high test MSE. The Ridge regression variant achieves a much better performance at higher levels of $\lambda$, which penalizes the $\beta$ term's size.