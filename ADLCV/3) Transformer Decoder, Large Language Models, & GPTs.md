
# Decoder
Access to *entire encoded* context. It is **auto-regressive**; Output 1 token *at a time*!
Input; encoder output, plus a token:
- First token is usually a `<START>` token.
- Add the output (append) to input alongside the encoded input.
- Repeat until `<STOP>` or `<END>` token is generated.

Can't generate outside its ==vocabulary== (which is the tokens it can generate).

*Auto-regressive* models are slow, especially when training! **Solution**:
During training, feed the entire ground-truth tokens! Makes training stable, parallel and we can compute a loss batch in a single forward pass.

During **inference**: it *has to be* auto-regressive, which is *slow* and *must feed on its own predictions* (==exposure bias==)

## 1) Masked (Multi-head) Attention
Have a mask so we *only attend* previous positions! This allows computing predictions for *all positions **in parallel*** (==single forward pass==, ==teacher forcing==)!

Set future positions to $-\infty$ (this makes the softmax for between *future* and *past/present* tokens  practically 0)

*Only used in decoders*

## 2) Cross Multi-head Attention

Keys and Values are from *encoder*, but Query is from the *previous part of the decoder layer*
**Note**: It is the *last encoder layer*'s output used in *all decoder layer*'s

## 3) Linear + Softmax
**Output size**: vocabulary size

# Types of Models
Some transformer models exclude the decoder, when generation is not needed. Seen in:
- Text classification
- Sentiment Analysis
- BERT(s)

Some decoder only architectures:
- Most LLMs
- GPTs
- Next token prediction

Both:
- Seq-2-seq models (Machine Translation)

## Why Decoder Only?
They are *simple in architecture*, and simple *training objective*. Text in $\rightarrow$ text out! It's a shift from per-task *architecture* to a per-task *instruction* (==prompt==).

# LLMs
Designed for NLP tasks such as *language generation*. Trained with *self-supervised learning* on *vast amounts of text*.

## Masked Language Modelling

Mask a word in "Encoder style" models, to then try and predict said ==MASKED== word! BERT was trained this way.

## Tokenization
Tokens are ==not whole words==. We split *text* into *tokens* via ==tokenization==. 
==Pro== & ==Cons== of
- Character Tokenization
	- Can generalize to any new words that contain those characters (other languages)
	- Harder to train with, as a word must be predicted by all its characters! Narrower context window
- Word Tokenization
	- Easier to train and decode with than characters
	- Bigger vocabulary (need to contain the words), can't generalize easily outside its vocab.
	- Tokens contain semantic information
	- Supports longer sequence lengths!
- ==Sub-Word tokenization==:
	- *Preferred* & in practice the one that is used.
	- Combination of *Character* & *Word*-tokenization, with *sub-words* as well.
	- UTF-8 Encoding & Byte Pair Encoding


# Decoding Strategies

***Temperature scaling***: affect the output probabilities, to be more *similar* with *higher temperatures*:
$$ P(w_i) = \frac{\exp(z_i/T)}{\sum_k\exp(z_k/T)}, $$
which is disabled when $t=1$, and ==uniform distribution== as $T\rightarrow\infty$, and towards a ==Dirac's delta distribution== as $T\rightarrow 0$!

***Greedy***: simplest decoding probability, pick word with highest probability (==deterministic==): 
$$w_t = \arg\max_w p(w|w_{1:t-1}),$$
but can easily get *stuck in loops* - a local minima of sorts.

***Beam search***: reduce risk of missing high-probability word sequences. **Keep most likely** hypothesis at each time step $t$. Based on `num_beams` parameter. Now pick based on the sequence probability: $\prod_t p(w_t|w_{1:t-1})$

Works well when generation *length* is *predictable*; can *suffer from repeating*. Human language ***does not*** follow most-likely words - it is not predictable.

***Multinomial Sampling***: randomly pick next word according to conditional distribution:
$$ p(w_t) = p(w_t|w_{1:t-1}), $$
can generate incoherent speech! **Tricks to circumvent**:
- *Increase* likelihood of high probability words
- *Decrease* likelihood of low probability words

Use *temperature scaling* with $T<1$, can achieve this

**Other strategies:** `TopP` & `TopK`
- **Top-K**:
	- Keep the top *k* tokens after softmax, set other tokens to 0, re-normalize distribution, and sample from this
- **Top-P** (==preferred==):
	- Sort tokens by probability
	- Calculate cumulative sum
	- Stop once cutoff $p$ (example would be 0.9)
		- Easier to reason about than the $k$ parameter in *Top-K*!

==INSERT PICTURE OF OVERVIEW FROM SLIDES HERE==



