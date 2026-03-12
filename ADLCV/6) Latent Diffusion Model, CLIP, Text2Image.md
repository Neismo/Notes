
# Scaling Diffusion Models
Sampling is *super slow* when in *high resolutions* (1000 steps, several minutes); *scales poorly*.
Images contain a lot of **redundant** or **imperceptible** details in pixels.

To scale we do it in a *latent*/*representation* space
# Latent Diffusion models (Stable Diffusion)

![[StableDiffusion.png]]

Model that combines *semantic compression* and *perceptual compression*. So a combination of *auto-encoders* with *diffusion models*. Work in *latent space*. 

Train VAE or Auto-encoder of any kind with a latent space (bottleneck so $d\ll D$).
With the trained encoder and decoder, we train a diffusion process in the latent space, and denoise also with a denoising U-net.

This allows *much faster* sampling as we are working in a much lower resolution for the Diffusion process.

## Text 2 Image (Txt2Img)

Allow for freedom in prompts, so we have a concatenation with $\tau_{\theta}$ as seen above, using *cross-attention*. For this we can use CLIP for example for the representations. The *queries* are from the latent space image, and the *keys* and *values* from the $\tau_{\theta}$.

Train with text, but randomly drop the text conditions with some probability. Then it can work both *conditionally* and *un-conditionally*.

In sampling we *extrapolate* between the conditioned and un-conditioned.
$$\tilde{\epsilon}_{\theta}(z_{t},t,c)=\epsilon_{\theta}(z_{t},t,\emptyset)+w[\epsilon_{\theta}(z_{t},t,c) - \epsilon_{\theta}(z_{t},t,\emptyset)],\quad w\in[0,1]$$

Text goes to concatenation with $Z_{T}$, whereas representation go to $K,V$ computations in *cross-attention*

## Editing Images (Img2Img)
Encode input image, add noise up to some $k$; in the middle usually. Then *denoise* using the new text-prompt as in Txt2Img

## DDIM: Denoising Diffusion Implicit Models

DDIM:
- Same training procedure
- Remove stochasticity (the *random* noise $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$)
- Gain speed (10x to 50x)
- Reconstruct with *low error*

So in DDIM we have a *deterministic* sampling process. Has a combined variant with a $\eta$ parameter controlling stochasticity (1 = classic DDPM, 0 = DDIM).

==INSERT IMAGE OF SUMMARY COMPARISON BETWEEN DDIM AND DDPM==
### DDIM Inversion

As DDIM is deterministic, we can find the latent $z_{T}$ that corresponds to a real image $x_{0}$ directly
$$ x_{t+1} = \sqrt{ \bar{\alpha}_{t+1}}\hat{x}_{0} + \sqrt{ 1-\bar{\alpha}_{t+1} }\ell_{\theta}(x_{t},t) $$

$$ \hat{x}_{0} = \frac{x_{t} -\sqrt{ 1-\bar{\alpha}_{t} }\ell_{\theta}(x_{t},t)}{\sqrt{ \bar{\alpha}_{t} }} $$

### Pivotal Tuning
We need to store the intermediate noisy steps from our input image, and then we want to train something small that pushed the process towards *making* those intermediate steps (loss); otherwise there is a drift, and the inversion is not exact/close.

## CLIP (Contrastive Language-Image Pretraining)

![[CLIP.png]]

Text and Image pairs, learn meaningful representations of *both*. **Contrastive loss**: match captions with image representation. Diagonal should match (max cosine in diagonal), off-diagonal should be *pushed apart* (minimize).

### Zero Shot Classification
CLIP show *zero-shot classification*; meaning it can *classify* without being trained to do so. For example with CLIP, I can have an image, and then get pairs of all kinds of "A photo of a X", where X is dog, cat etc.

Pick the one with largest similarity; it now classifies.
# Controlling Image Generation

# Self Supervised Learning
Model is trained to predict a naturally occuring signal in the raw data; not from human labeling.

# Exercise 2.3
- Visualize Cross Attention
- Implement DDIM sampling
- Implement DDIM inversion
- Implement Null-text Inversion
- Edit images