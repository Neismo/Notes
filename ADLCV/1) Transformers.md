
![[TransformerArc.png]]


# Sequence-2-Sequence

Takes one sequence of information $x$ and outputs to another sequence $y$
- NLP: Translation, continuation etc.

NLP a bit behind other fields earlier:
- ConvNets / vanilla use fixed-size input/output
- Some tasks require sequence of inputs/output vectors
- Limited receptive field

RNN were introduced for partial sequences of input and output (carry information forward through hidden unit):
- $y_i = f(Wh_{i}+b),\quad h_i=g(W[h_{i-1};x_t]+b)$
- Backpropogation required to take account into a sum over all paths from loss to weights; problematic:
	- Sequence processing *can't* be done in parallel
	- Influence is quickly lost except in the next (few) steps.
	- Very hard to train / scale up! Vanishing/Exploding gradients.
- Workaround (not solution) was:
	- Bi-directional models
	- LSTMs (Long Short Term Memory)

How were they overcome:
- What if every word could relate to any other word regardless of distance, directly?
	- Attention module from "Attention is all you need"

## Transformers & Attention
Simplest representation: $input\rightarrow encoder \rightarrow decoder \leftarrow \text{curout},\quad decoder\rightarrow output$
Encoder:
- **input**: *tokens* (works, characters, in-between etc)
- **output**: representations of said tokens (*context*)
- Can run in *parallel*
Decoder:
- Access to *entire* encoded context
- **auto-regressive**: input is previous currently seen context (prediction 1 token at a time)

### Encoder
- Input Embeddings
- Positional Embeddings 
- $N$ Transformer Encoder Layers (above processing only done the first time!)
	- Multi-head Attention
	- Add & Norm
	- Feedforward
	- Input is previous layer's output

**Self Attention**: each token attends to all tokens, including itself, in the sequence:
- *example*: "The animal didn't cross the street because it was too tired"
	- Here '*it*' may refer mostly to "animal"; in reality, it has different *weights* to each token in the sequence
- Calculated in *parallel*
- *Query, Key, Value* matrices:
	- Match *Query* to ==all *Keys*==, producing a *Value* that matches best (*dot-products*)
	- From embedding to $query_i$, $key_i$ and $value_i$ vectors
		- Each has a weight matrix: $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ from which they are produced
	- Score is then calculated with $query_i$ to ==each== $key_k$ for all $k$ in the vectors (dot-product)
	- Divide by $\sqrt{d_k}$ (usually 8)
	- Softmax
	- Multiply with $value_i$ producing $z_i$
		- All this can be done with *stacked* vectors! (So stack all queries, keys and values)
		- $\mathbf{Z} = \text{softmax}(\frac{\mathbf{Q}\times \mathbf{K}^\top}{\sqrt{d_k}})\times \mathbf{V}$
- *Multi-Head Attention*:
	- Multiple attentions - like multiple channels in CNNs:
	- By the end of each attention, concatenate resulting $Z_i$ from each attention block and multiply with final weight matrix $W_O$
	- Each head looks only at part of the embeddings - but still parts of all tokens - so not as computationally expensive.
	- Each block has their own $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ matrices (so 1st would be $\mathbf{W}_{Q,1}, \mathbf{W}_{K,1}, \mathbf{W}_{V,1}$)
- Attention is *equivariant* due to matrix multiplication is permutation equivariant
	- *Invariant*: transformations to input do *not* change output
	- *Equivariant*: transformations to input, is a similar transformation in output: $f(g\cdot x)=g\cdot f(x)$!
	- This is a problem as positional information is *not* encoded
		- That is why positional encodings are added

# Exercises (3rd February):
Classification from Transformer Encoder:
1) Remove decoder from Transformer
2) Add avg-pool (or similar) to Encoder output
3) Add simple layer as "decoder"
4) EXTRA) Add a \[CLS\] token and use its output to the linear layer (no avg-pool/ignore other outputs!)
	1) Take this tokens embedding, ignore all other embeddings and classify based on this (with the linear layer)
5) EXTRA) Learnable PE
IMDB movie reviews --> sentiment classification