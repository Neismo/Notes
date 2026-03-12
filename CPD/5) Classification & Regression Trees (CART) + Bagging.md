
# Regression Trees
A decision tree that mimics series of decisions - but is data-based! This can be knots, where each region is then based on the *mean* of the data in the box;

Fitting such knots:
- Place one knot after one another
- Place such that it minimizes an error

Each split can be represented by a *parent node* and two *child nodes*.

**Multiple input variables**: loop over variables, inner loop over splits, innerst do eval of the split - when out choose best split.

**What is a good split**: based on RSS (residual sum of squares): $\sum_{i\in I}(y_{i}-\hat{y}_{i})^2$, which is minimized when $\hat{y}$ is the mean.
**How many splits**: for $n$ data points, we have $n-1$ splits between the data points (mid points). Compare RSS for each of those splits - choose one with lowest.

## Categorical variables:
Empty groups not allowed, so the number of splits are:
- A binary input variable has only *1* split (man or woman), a trinary input variable has *3* possible splits:
	- Example {apple, banana, orange}:
		- (apple), (banana, orange)
		- (banana), (apple, orange)
		- (orange), (apple, banana)


**How large does the tree grow**: stop splitting based on number of data points in split (e.g., 10).


## Algorithm
![[classificationTreeAlgo.png]]

Another approach is to allow the tree to grow large and then ==prune==! Prune node whose *sub-tree* give **smallest** *per-node* reduction in RSS - divide the *reduction* by number of *terminal nodes* minus 1.

Doing a lot of pruning, *when do we stop*? Use **independent test-set** to validate each sub-tree and choose based on that.

## Classification Trees
Assigns class instead of a constant value. Different criteria but similar to *regression trees*. Instead of using the *mean* we assign to the *majority class* of the observations in the node.

Out error measure is:
- Miss-classification rate
- Gini Index
- Cross-Entropy

All a favor a split that increases *purity*
### Purity of Nodes
In a specific node, representing a region $R$ with $N$ observations, let
$$ \hat{p}_{k} = \frac{1}{N}\sum_{x_{i}\in R}\mathbb{1}(y_{i}=k) $$

Classify observations in the node to class
$$ K = \arg\max_{k}\hat{p}_{k}, $$
The measure *impurity*, $Q$, in the node:
- **Miss-classification error**: $Q  = \frac{1}{N}\sum_{x_{i}\in R}(y_{i}\neq k)=1-\hat{p}_{k}$
- **Gini Index**: $Q = \sum_{k\neq k}\hat{p}_{k}\hat{p}'_{k} = \sum_{k}\hat{p}_{k}(1-\hat{p}_{k})$
- **Cross-Entropy (GE)**: $Q = -\sum_{k}\hat{p}_{k}\log(\hat{p}_{k})$
*Preference is*: CE or Gini

We *weight* by the number of nodes in the split *N*, and we split based on **minimizing** the following
$$ N_{\text{left}}Q_{\text{left}}+N_{\text{right}}Q_{\text{right}} $$

**Gini** for building tress, **miss-classification error** for pruning trees.

### Missing Variables
Introduce an extra category for "missing".
Surrogate variable.

# Bagging
Ensemble method
Many models on bootstrapped samples
Output from all models aggregated to one model (average)

*bootstrap*: with replacement

1) Make 'B' bootstraps
2) For each bootstrap 'b' do
	1) Fit a model to the bootstrap, and make prediction
3) Bagging estimate is average of B predictions
$$ \hat{y}_{\text{bagging}} = \frac{1}{B}\sum_{b=1}^B\hat{y}_{b} $$

Bagging reduces variance, but does nothing to bias.

# Exercises
{1,2,3,4}:
- {1}{rest}
- {2}{rest}
- {3}{rest}
- {4}{rest}
- {1,2}{rest}
- {1,3}{rest}
- {1,4}{rest}

{1,2,3,4,5}:
- {1}{2,3,4,5}
- {2}{1,3,4,5}
- {3}{1,2,4,5}
- {4}{1,2,3,5}
- {5}{1,2,3,4}
- {1,2}{3,4,5}
- {1,3}{2,4,5}
- {1,4}{2,3,5}
- {1,5}{2,3,4}
- {2,3}{1,4,5}
- {2,4}{1,3,5}
- {2,5}{1,3,4}
- {3,4}{1,2,5}
- {3,5}{1,2,4}
- {4,5}{1,2,3}