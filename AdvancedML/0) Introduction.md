l
# Why Deep Generative Modeling?

We want models to be uncertain in regions where it did not have data. For example in a 2D example with two regions of data points (different classes), even if we get a decision boundary, if we see a data point far from what has been observed, we want to have a low marginal probability $p(x)$ to weight down the certainty.

So the claim is that considering $P(x,y)=P(y|x)p(x)$ has clear advantages

## How can we formulate it?

==Four categories== of *(deep)* generative models:
- Auto-regressive Generative Models (ARMs)
- Flow-based Models
- Latent Variable Models
- Energy Based Models
- Score Based Models

*(deep)* because neural networks are not strictly required for modelling, but will be the focus of the notes.

![[DeepGenerativeModeling.png]]


### ARMs

The distribution of $p(x)$ is represented in an auto-regressive manner:
$$ p(\mathbf{x}) = p(x_0)\prod_{i=1}^Dp(x_i|\mathbf{x}_{<i}) $$
Where $\mathbf{x}_{<i}$ denotes all $\mathbf{x}$'s up to index *i*. Modelling *all* the conditional probabilities are not realistic, but we can use *causal convolutions* in audio and imaging.

### Flow Based Models
We can also express it with by a transformation with an invertible transformation *f*:
$$p(\mathbf{x})=p(\mathbf{z}=f(\mathbf{x}))|\mathbf{J}_{f(\mathbf{x})}|,$$

where $\mathbf{J}_{f(\mathbf{x})}$ denotes the Jacobian matrix. *f* can be parameterized by a neural network, but not any, as we need to be able to calculate the Jacobian. 

### Latent Variable Models
In this approach we assume a lower-dimensional latent space and a following generative process:

$$\begin{align}
\mathbf{z} &\sim p(\mathbf{z}) \\
\mathbf{x} &\sim p(\mathbf{x}|\mathbf{z})
\end{align}$$
Widely known is **pPCA** (probabilistic principle component analysis) where both are following Gaussian distributions, and the dependency between the two is linear.

Another is **VAEs** (Variational Auto Encoders) with arbitrary distributions. To make inference tractable, neural networks parameterize the distributions, and variational inference is utilized to approximate the posterior.

So far **ARMs**, **Flow Based Models** and **pPCA / VAEs** all work with *log-likelihood functions*, closely related to the *KL-divergence*. Another approach would be a *adversarial approach* with a discriminator $D(\circ)$ to determine the difference between real and synthetic data.