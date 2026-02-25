
# Generative Models
- Based on probabilistic density models
- Sample based (parametric)
 
 Examples of generative models are:
- VAEs
- GANs
- Normalizing flow models

No dimensionality reduction in the latent space unlike VAEs.

## Bird's Eye View
Diffusion models sample from a ==latent space== which follows a distribution we know; often called *noise*, but not in the *statistical sense*.
- Encode input by adding noise over *T* time steps
- Learn to decode each step
- Noisy images are time-dependent variables

**Forward process**: add *noise*
**Reverse process**: denoising

![[BirdsEyeViewDiffusion.png]]

## Forward Process
Markov chain:
$$ q(\mathbf{x}_{T:1}|\mathbf{x}_{0})=\prod_{t=1}^Tq(\mathbf{x}_{t}|\mathbf{x}_{t-1}),\quad q(\mathbf{x}_{t}|\mathbf{x_{t-1}}) = \mathcal{N}\left(\mathbf{x}_{t}|\sqrt{ 1-\beta_{t} }\mathbf{x}_{t-1},\beta_{t}\mathbf{I}\right), $$

Where $\beta_{1},\cdots,\beta_{T}$ follow a *linear scheduler* or *cosine scheduler*.

## Reverse Process
A Markov chain of learnable Gaussian transitions, utilizing deep learning model for representation:

$$ p_{\theta}(\mathbf{x}_{0:T}) :=p(\mathbf{x}_{T})\prod_{t=1}^Tp_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}),\quad p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) = \mathcal{N}\left( \mathbf{x}_{t-1}|\mathbf{\mu}_{\theta}(\mathbf{x}_{t},t),\Sigma_{\theta}(\mathbf{x}_{t},t) \right),$$
with $p(\mathbf{x}_{T})=\mathcal{N}(\mathbf{x}_{T}|\mathbf{0},\mathbf{I})$. How to learn $p_{\theta}(\circ)$?
### Learning denoising
We want to learn the noise that was added from $x_0$ to $x_T$. The close form looks like:
$$ q(\mathbf{x}_{t}|\mathbf{x}_{0}) = \mathcal{N}(\mathbf{x_{t}}|\sqrt{ \bar{\alpha}_{t} }\mathbf{x}_{0},(1-\bar{\alpha})\mathbf{I}), $$
which we write as:
$$ \mathbf{x}_{t}(\mathbf{x}_{0},\epsilon) = \sqrt{ \bar{\alpha}_{t} }\mathbf{x}_{0} + \sqrt{ 1-\bar{\alpha}_{t} }\mathbf{\epsilon}, \epsilon \sim\mathcal{N}(\mathbf{0},\mathbf{I}). $$

This predictor $\epsilon_{\theta}$ will follow a U-net as a backbone, with the time step $t$ as input in various places (either as a channel or as positional encoding on top of the input in various layers)

As we want to learn the noise, the loss in practice is the RMSE loss between predicted and actual noise:
$$ L_{simple}(\theta):=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{\epsilon}}[||\mathbf{\epsilon}-\mathbf{\epsilon}_{\theta}\left(\sqrt{ \bar{\alpha}_{t} }\mathbf{x}_{0} + \sqrt{ 1-\bar{\alpha}_{t} }\mathbf{\epsilon},t\right)||^2] $$