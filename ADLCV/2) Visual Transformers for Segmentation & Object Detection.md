
1) Transformers beyond NLP
2) Vision Transformers (ViT)
3) Application in Object Detection & Image Segmentation

# Image Classification
Conventionally with **CNN**: image to class label through convolutions.
*Idea*: go through transformer encoder.

**Initial idea**: start from CNN architecture; attention *inside* CNNs between resnet-blocks; features *as* tokens.
**Another idea**: *local attention*; use kernels _center_ as *query*, and *keys/value* are each element in the *receptive field*. Was hard in practice, and marginally better only.

**Pixel Transformers**: treat image as pixel values, flatten them, and feet input to standard transformer; *main problem*: memory usage due to image size (insanely high memory usage)

**Vision Transformers**: first successful usage of transformer encoders for images.
1) split image into $16\times 16$ patches, where each is a "token" (disjoint); $[3\times 16\times 16]$
2) Flatten patches
3) Linear project the patches to *D*-dimensional embedded vector patches; $[D]$; 
4) Add position embedding; learned 1D vector *per position*
	1) Also add CLS token to 0th position for an extra learnable embedding (pre-pended *after* projection).
5) Classic transformer encoder ($\times L$ layers) with the *embedded patches*
6) MLP head to $C$ classes
7) Classify

**In practice**:
1) Input image is $224\times 224$
2) Divide into $14\times14$ grid of $16\times16$ pixels (or reverse)
3) Require more data than ResNet for pre-training before they perform better
	1) Generally, CNNs have *inductive bias*, which assumes local features are correlated; transformers have to learn this for images.
Similar *attention visualizations* as with text; with respect to the *output class*; conceptually similar to *saliency maps*.

## Positional Embedding
- Models learn to *encode distance* (close patches have similar positional embeddings)
- Row/column *structure appears* (Shared rows/columns share similar positional embeddings as well)
- *Sinusoidal* tendencies appear in larger grids
- *Position embedding* learn *2D image topology*


## Inductive Bias
Assumptions / Preferences / priors in algorithms used to predict outputs for data; allows *generalization* beyond training data.

Examples:
- *Linear Regression*
- *Decision Trees*: favor's shorter trees
- *CNNs*: spatial locality and translation invariance

ViT:
- Global attention / connectivity
- Permutation invariance

### CNN vs VIT:
- CNNs inductive bias (local features correlated) *
- Patches are arbitrary
- CNN have some shared weights in channels
- ViT can *lose information* / *lost structure*
- CNN is translation equivariant, ViT is not, requires more data *
- ViT has *global attention* $\rightarrow$ scales better (can capture information across the image, requires many convolutions in CNN to achieve this with receptive field)

## Data Efficient Image Transformer (DeiT)
Problems with ViT were:
- required massive datasets
- Poor performance on IN-1K
- Lacks inductive bias

DeiT solutions:
- Train ViT on IN-1K only
- Strong data augmentation + regularization
- Knowledge distillation from CNN teacher
	- ResNet50 as *teacher*
	- ViT is *student* 

### Knowledge Distillation
A larger *teacher* network, that has been trained, and a smaller *student* network. Transfer as much *distilled* knowledge from *teacher* to *student* models. 

Compare predictions from teacher to student to generate loss and compare to *ground truth*.
- 1st term loss: Cross Entropy Loss between *student* and *ground truth*
- 2nd loss: *soft divergence*: match KL divergence between *student* and *teacher* + *hard divergence* for mimicking predictions between *teacher* and *student*.

## Hierarchical Swin Transformers

Swin Transformer in the beginning, starting with *much smaller* patches; then combine in higher layers.

*Transformers*:
- Quadratic Complexity, O(n^2) attention, single scale

*Swin Improvements*:
- Window attention in *neighbourhood* more similar to CNNs (+ more computationally efficient)
- Shifted attention in *next layers beyond* so window attention is *shifted/different*
- Patch merging, multiple scales
- Higher resolution

*Internally*:
- Start with $2\times2$ patches, then merge $2\times2$ neighbourhoods for effective $8\times8$; continue like this ends up at $32\times32$ patches.

# Object Detection with Transformers

DETR (before ViT):
- Image to CNN to get features (+ positional encoding on top of features)
- Features through Transformer Encoder-Decoder to a set of box predictions
	- Input to decoder are *object queries* similar to *CLS token* (also encoder output of course)
	- Transformed to *output embeddings* that are independently decoded into box coordinates and class labels by a MLP layer.
	- In practice, each *query* specialize on certain areas and box sizes after training. 
- Bipartite matching loss
	- End to End training!
	- Chose $pred\rightarrow GT$ and choose such that each GT only has 1 source from predictions.
		- One-to-one matching compared to NMS

# Image (Semantic) Segmentation with Transformers

SETR:
- Same as ViT, but with *decoder*
- Decoder is CNN architecture

SegFormer:
- More efficient; similar ideas to *Swin* with different patch sizes.
- Efficient Self-attention
- Smaller, better decoder

MaskFormer:
- Per-pixel was not enough
- Structure:
	- Pixel-level-module: *input features*
	- Transformer module: with decoder similar to DETR
- Mask2Former:
	- Masked Attention


# Summary
- Application across different tasks: classification, object detection & segmentation
- Benefits: speed, more hardware friendly
- ViTs > CNNs