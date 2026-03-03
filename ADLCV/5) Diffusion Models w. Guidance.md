
Combining generative image models with additional input information

## Classifier Guidance for DDPM
Assume a ==trained, unconditioned DDPM== and images ==with labels==. For this we require the gradient with *respect to the image* space $\nabla_if$.

![[ClassiferGuidedDDPM.png]]

See how the mean we're sampling from is pushed into the direction of the gradient of the classifier with respects to the image's class - $s\Sigma \nabla_{x_{t}}\log p_{\phi}(y|x_{t})$. 

The scale factor $s$ determines the *step size* and controls reconstruction accuracy, but penalizes variation: This is because $s\cdot \nabla_{x_{t}}\log p(y|x_{t})= \nabla_{x_{t}}\log p(y|x_{t})^s$, so it pushes the probability toward modes of the distribution.

The classifier is trained on *noisy* variants of the images. Input to the classifier also requires the time step *t*.

## Classifier Free Guidance
Having a new classifier makes the training process more complicated. Can it be achieved without it? Use ==conditioning== to generate images:
![[GuidanceFreeDDPM.png]]
or ==Classifier-free Guidance== which trains it jointly with the diffusion model:
![[Classifer-free Guidance.png]]
New additions here:
- Condition on *class*; additional input
- Joint training of conditional and unconditional model
- Rest is just another notation
- **Required**: one-hot encoding of the classes.
- **Guidance strength**: not shown in this algorithm, but usually a trade-off between conditional and *un*conditional training with parameter $w$.

### Frechet Inception Distance

$$ d_{F}(\mathcal{N}(\mu,\Sigma), \mathcal{N}(\mu',\Sigma'))^2 = ||\mu-\mu'||^2_{2}+\text{tr}\left( \Sigma + \Sigma' - 2 \left( \Sigma^{1/2}\cdot \Sigma'\cdot \Sigma^{1/2} \right)^{1/2} \right) $$
- Assumes normal distribution of data
- Based on features from for example *InceptionNet*.

### Validating Generative Models
*Precision* and *Recall* for distributions:
![[PrecisionRecallDistributions.png]]

# Counterfactual Explanations
*How should input change to obtain a different classification*

Most have in common:
- ==Data Comparability term== that keeps counterfactuals close to the real data (*the diffusion model*)
- A term that drives your counterfactual towards the ==prediction interest==. (*classifier guidance*)
	- **Issue**: requires a classifier trained with noisy images, but an off-the-shelf one is not.

## First Approach: iterative denoising
**Idea**: take image of what you want to change, add some noise (but not too much) and then generate with DDPM the de-noised of that and push it towards what you want. Do this at every denoising step:

![[GuidedStepCounterfactuals.png]]
![[Counterfactuals.png]]

- Every denoising step also has a classifier gradient term
- **very expensive**; denoising at *every step* with DDPM. Due to the gradient calculation.

We can do a reparameterization trick to quickly generate training examples:
$$ x_{0} = \frac{x_{t}(x_{0},\epsilon)-\sqrt{ 1-\bar{\alpha}_{t} }\epsilon}{\sqrt{ \bar{\alpha}_{t} }} $$

## Universal Guidance
Can be used to generate images matching a prompt *c* of any modality. Through a loss $I(c,f(x))$
- *forward:* taking gradient steps w.r.t. the noisy image $z_t$ by translating it to an estimate of $z_{0}$
- *backward*: The reverse of forward.

Example for segmentation masks and dogs:![[UniversalGuidanceExample.png]]

