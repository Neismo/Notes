
**Idea of self-supervised learning**: model is trained to predict a *naturally occuring part* of the *raw data* without human annotations.
**Different from unsupervised**: as *it is told* what to predict; just not by humans.

2 stages; *pretext* and *downstream*.
- *Pretext Example*: auto-encoders with reconstruction
- *Downstream Example*: encoder used for feature extraction (classification downstream)

3 big groups of pretext tasks:
- **Generative** - predict part of the signal: auto-encoders, GANs, Colorization
- **Discriminative** - predict something about the input signal: Contrastive (CLIP), Clustering, Rotation, Context Prediction.
- **MultiModal** - use additional information to f.x. image: 3D, sound, language, video

*Generative*
- **Colorization**: take input image, greyscale it, and learn the model to do that self-supervised. Hope models learn structure of image and coloring techniques.

*Discriminative*
- **Context Prediction**: learn context of parts of the image by other parts (patches in the image)
- **Deep Clustering**: run images through (2) random CNN (1), then K-NN on feature vectors. Use cluster assignments as pseudo-labels and use the CNN to predict cluster assignments; repeat from running images through the CNN (2).
- **Invariance to Data Augmentations**: we want to learn same image with various data augmentations to have same representation. 
	- Input augmented image to CNN, and have it predict which original image it was (N-way classification); poorly scaleable (one class per image)
	- **Contrastive Learning**: N-images; create two augmentations of these images (patch of image). Feed into each patch into CNN to extract features. Match every patch with every other patch. Positive pairs are the two patches from same image. Positive-negative pairs. Maximize similarity between positive, push negative apart.



**DINO**: student and teacher networks; the teacher are updated with EMA from student. Same architecture but different parameters. Student updated as usual with SGD