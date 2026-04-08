
# Introduction

Graphs are ==ubiquitous data structures==; a universal language for describing complex systems. Its power is in *relationship between points*. 

## What is a graph
>[!note] **Graphs**
>Graphs are *formally* denoted by $\mathcal{G} = (\mathcal{V},\mathcal{E})$ and consists of a set of edges $\mathcal{E}$ between the set of nodes $\mathcal{V}$.  A directed edge from node $u\in \mathcal{V}$ to $v\in \mathcal{V}$ is denoted as $(u,v)\in \mathcal{E}$,

**Simple Graphs**: At most one edge between nodes; this means there is no concern of *direction*. Formally: $$ (u,v)\in \mathcal{E} \leftrightarrow (v,u)\in \mathcal{E} $$
**Adjacency Matrix**: a matrix in  $\mathbb{R}^{|\mathcal{V}|\times|\mathcal{V}|}$ which stores the connection between nodes; 1 denotes a *edge*, and 0 denotes no edge. $\mathbf{A}[u,v]=1\rightarrow(u,v)\in \mathcal{E}$. $\mathbf{A}$ is symmetric for *simple graphs*; otherwise it may not be (==directed graphs==). Can also contain *weighted* edges, where we allow the values in $\mathbf{A}$ to contain other values than 1 or 0.

### Multi-Relational Graphs
We can also have ==different type of edges== denoted with $\tau$. Here we *extend* the notation of edges to be $(u,\tau,v)\in \mathcal{E}$. Adjacency matrices are then per type: $\mathbf{A}_{\tau}$. We can combine these in a *Adjacency Tensor* of size $\mathbf{A}\in \mathbb{R}^{|\mathcal{V}|\times|\mathcal{R}|\times|\mathcal{V}|}$, where $\mathcal{R}$ are the set of relations. Two important sub-set of graphs.

**Heterogeneous Graphs**: in these graphs, nodes also have types and are ==disjoint== such that $\mathcal{V}=\mathcal{V}_{1}\cup \mathcal{V}_{2}\cup\cdots\cup \mathcal{V}_{k}$ such that $\mathcal{V}_{i}\cap \mathcal{V}_{j}=\emptyset$ for all $i,j$. In these, edges of types usually only connect certain node types.

**Multiplex Graphs**: in these we assume the graph can be structured in *k* layers. Every node belong to a layer, each layer is a *unique* relation, corresponding to an intra-layer edge type for that layer. Inter-layer edges can exist as well.
> **Example**: For instance, in a multiplex transportation network, each node might represent a city and each layer might represent a diﬀerent mode of transportation (e.g., air travel train travel). Intra-layer edges would then represent cities that are connected by diﬀerent modes of transportation, while inter-layer edges represent the possibility of switching modes of transportation within a particular city.

### Feature Information
*Attributes* or *features* can also exist for graphs. Often done at *node-level* and we have that $\mathbf{X}\in \mathbb{R}^{|\mathcal{V}|\times m}$, so $m$ features. In the *heterogenous graphs* we assume each node type has its own set of attributes.

## Machine Learning
ML with graphs is not much different; we have supervised and unsupervised learning. We will see that. Here are some classical problem types

### Node Classification (Semi-supervised)
Suppose a large graph network, and that many of these *users* may be bots. Manually classifying them would be an impossible task, so here we want to do ==node classification==! The goal is to predict the label $y_u$ for all nodes $u\in \mathcal{V}$, when given the true labels for a subset $\mathcal{V}_{\text{train}}\subset \mathcal{V}$.

Nodes are **NOT I.I.D** which is a classic assumption. In fact, they are very much connected, that is why we model it as a graph. We can exploit this *homophily*: tendency for nodes to share attributes.

### Relationship Prediction (Semi-Supervised)
When we are missing relationship information, can we predict that? Goes by many names: *link prediction, graph completion* or *relationship prediction*. Given nodes and an **incomplete** set of edges $\mathcal{E}_{\text{train}}\subset \mathcal{E}$, the goal is to infer the missing edges $\mathcal{E} \backslash \mathcal{E}_{\text{Train}}$.

### Clustering & Community Detection (Unsupervised)
We want to do some form of clustering; examples are collaboration graphs from citations of Google Scholar, that connects two authors if they have co-authored a paper together. We expect a *community* structure, where there are clusters around important attributes.

Given only the input graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ can we infer this? Used for uncovering for example fraudulent groups of users in financial transaction networks.

### Graph Classification, Regression & Clustering
Done over the *entire* graphs. Perhaps we want to regress the toxicity of a molecule based on its graph? Or classify the maliciousness of a program based on its syntax graph.

In this setting we usually have *multiple different graphs* and to make *independent predictions* specific to each graph. These are funnily enough the most similar to traditional ML.

# Background & Traditional Approaches

Traditional approaches before deep learning used heuristics. Extracting some features or statistics. These features are then used as input to for example *logistic regression models*. First some of *node-level* features and statistics, then generalized to *graph-level*.

## Node Level Statistics & Features

![[MarriageTree.png]]

Consider Above tree, as the example throughout this part. These are political marriages, and illustrated the *rise of power for the Medici family*. What feature or statistics from ML can be used to infer this.

### Note degree
>[!note] Note Degree
>The *degree* $d_{u}$ of a node $u\in \mathcal{V}$ is the counting of number of edges incident to a node - in other words the number of *neighbours*. $$ d_{u} = \sum_{v\in \mathcal{V}} \mathbf{A}[u,v]. $$
>Note, in directed graphs, there is a ==different *degree*== associated with *outgoing* and *ingoing*.

In our example, this is a good feature to distinguish the Medici family from the others; it has the *highest* degree in the graph (6). What about other more discriminative features?

### Node Centrality
Neighborhood is not necessarily an indicator of importance. One example is the *eigenvector centrality* measure. It takes into the account a neighbours importance as well
>[!note] Eigenvector Centrality
>Takes into account how important neighbours are as well! $u\in \mathcal{V}$: $$ e_{u} = \frac{1}{\lambda}\sum_{v\in \mathcal{V}}\mathbf{A}[u,v]e_{v},\quad \forall u\in \mathcal{V}, $$
>where $\lambda$ is a constant. In standard vector notation with $\mathbf{e}$ as the vector of node centralities we get the eigenvector equation for the adjacency matrix
>$$ \lambda \mathbf{e} = \mathbf{A}\mathbf{e}. $$

### Clustering Coefficient

Degree and centrality are clearly useful; another is the clustering coefficient, measuring the proportion of closed triangles in a node's local neighbourhood. 

>[!note] Local Clustering Coefficient
>The ==local variant== is measured as follows: $$ c_{u} = \frac{|(v_{1},v_{2})\in \mathcal{E} : v_{1},v_{2}\in \mathcal{N}(u)|}{\begin{pmatrix} d_{u} \\ 2 \end{pmatrix}}, $$
>where $\mathcal{N}(u)=\{ v\in \mathcal{V}|(u,v)\in \mathcal{E} \}$. The denominator is how many pairs in the neighbourhood.
>A measure of 1 means all neighbours of $u$ are also neighbours with one another.

In our example, the Peruzi node has a local coefficient of 0.66, whereas Guadagni has one of 0 - due to no triangles.

## Graph Level Features & Graph Kernels

Let's consider graph level features now; they fall under *graph kernel methods*. 

### Bag of Nodes
Simply an ==aggregation of node-level features & statistics==. Could be histograms or other summary statistics on the features of nodes. Entirely based on node-level information and can miss global properties.

### Weisfieler-Lehman Kernel
An idea is *iterative  neighborhood aggregation*. The steps are seen below.
>[!note] Weisfieler-Lehman Kernel
>1) First, assign an initial label $l^{(0)}(v)$ to each node. A good start would be the degree of the node.
>2) Iteratively assign a *new label* to each node by hashing the multi-set of the current labels within the node's *neighbourhood*. $$ l^{(i)}(v) = \text{HASH}(\{ \{ l^{(i-1)}(u) \forall u\in\mathcal{N}(v) \} \}), $$ where double-braces are signifying the multiset, and the hash function maps each unique set to a unique value.
>3) After K iterations of re-labeling, we now have labels $l^{(K)}(v)$ for each node that summarizes the structure of it's K-hop neighborhood. Histograms and other summary statistics can then be computed.

It can be used for isomorphism checks between two graphs. Run it for K iterations and check if both share the same labels. Not guaranteed to work.

### Graphlets & Path-Based methods

## Neighborhood Overlap Detection

## Graph Laplacians & Spectral Methods

# Node Embeddings

Now for learning *node embeddings*. Encode low-dimensional summaries of graph position and structure.

## Encoder-Decoder Perspective
Framework of *encoding* and of *decoding*. First the encoder maps each node to a latent space, and the decoder reconstructs the neighborhood in the original graph.

### Encoder
Maps a node $v\in \mathcal{V}$ to an embedding vector $\mathbf{z}_{v}\in \mathbb{R}^d$; in its simplest form it is
$$ \text{ENC}:\mathcal{V}\to \mathbb{R}^d. $$
This would correspond to taking the node ID and generating node embeddings. In most work on node-embeddings, the encoder relies on a *loop-up table* of node embeddings denoted $\mathbf{Z}\in \mathbb{R}^{|\mathcal{V}|\times d}$:
$$ \text{ENC}(v)=\mathbf{Z}[v], $$
where $\mathbf{Z}[v]$ is the row embedding vector for node $v$. ==This is the focus for now==. Other more sophisticated methods can be used that utilize the features of local neighborhoods for example.

### Decoder
The decoders job is to reconstruct some of the features from the node embedding. This could be to predict the row $\mathbf{A}[v]$ or the neighbors $\mathcal{N}(v)$. The standard is the pairwise decoder
$$ \text{DEC}:\mathbb{R}^d\times \mathbb{R}^d\to \mathbb{R}^+. $$
This means the decoder operates of pairs of embeddings, and tries to predict some shared features between the two. 

Applying the decoder to the encoders output gives the ==reconstruction term==.
$$ \text{DEC}(\text{ENC}(u), \text{ENC}(v)) = \text{DEC}(\mathbf{z}_{u},\mathbf{z}_{v}) \approx \mathbf{S}[u,v], $$
where $\mathbf{S}[u,v]$ is assumed to be a graph-based similarity measure between nodes. It could be $\mathbf{S} = \mathbf{A},$ such that it needs to correctly reconstruct edges (or the lack thereof).

### Optimization
Standard is to optimize a loss:
$$ \mathcal{L} = \sum_{(u,v)\in\mathcal{D}}\ell(\text{DEC}(\mathbf{z}_{u},\mathbf{z}_{v}),\mathbf{S}[u,v]), $$
where $\ell:\mathbb{R}\times \mathbb{R}\to \mathbb{R}$ is a loss function measuring the *discrepancy* between the decoded similarity and the true values. Depending on the loss, it can be mean squared error or classification loss. The overall goal is to train the encoder and decoder so that pairwise relations can ==effectively be reconstructed==.

![[DecodersAndLossFunctionsGraphs.png]]

## Factorization Based Methods

The **inner product** methods employs an inner product for the decoder
$$ \text{DEC}(\mathbf{z}_{u},\mathbf{z}_{v}) = \mathbf{z}_{u}^\top\mathbf{z}_{v}, $$
in such a way that the similarity measure may be proportional to the dot product of the embedding vectors. The loss then follows a mean-squared error style
$$ \mathcal{L} = \sum_{(u,v)\in\mathcal{D}}|| \text{DEC}(\mathbf{z}_{u},\mathbf{z}_{v}) - \mathbf{S}[u,v] ||_{2}^2. $$

## Random Walk Embeddings
