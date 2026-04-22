
# GNR

Usual deep learning does not apply like CNNs to graphs.

## Neural Message Passing
We continue with *message passing*:
$$ \begin{equation}
\mathbf{h}_{u}^{(k+1)} = \text{UPDATE}^{(k)}\left( \mathbf{h}_{u}^{(k)}, \text{AGGREGATE}(\{ \mathbf{h}_{v}^{(k)}:v\in\mathcal{N}(U) \}) \right)\tag{Message Pass}
\end{equation} $$
here UPDATE and AGGREGATE can be neural networks as long as they are differentiable. At $k=0$ we simply have the nodes input features. After some iteration we have the embeddings:

$$\begin{equation}
\mathbf{z}_{u} = \mathbf{h}_{u}^K,\forall u \in \mathcal{V}.
\end{equation}$$

As they work on *sets* they are by definition permutation invariant (which is required).

>[!note] Node Features
>Can be shallow embeddings as discussed previously. Can in case of no features be one-hot encoding so each node has its own unique *index* and then aggregate those in message passing.


### Basic GNN
$$ \mathbf{h}_{u}^{(k)} = \sigma\left( \mathbf{W}_{\text{self}}^{(k)} \mathbf{h}_{u}^{(k-1)} + \mathbf{W}_{\text{neigh}}^{(k)}\sum_{v\in \mathcal{N}(u)} \mathbf{h}_{v}^{k-1} + \mathbf{b}^k \right). $$

So its weights are per layer *k* and one for itself and one for aggregation, and $\sigma$ is a non-linearity. Usually with *self-loop* each node is also its own neighbor. Also sometimes it is
$$ \mathbf{m}_{\mathcal{N}(u)} = \sum_{v\in\mathcal{N}(u)}\mathbf{h}_{v} $$
$$ \mathbf{h}_{u}^{(k)} = \sigma\left( \mathbf{W}_{\text{self}}^{(k)} \mathbf{h}_{u}^{(k-1)} + \mathbf{W}_{\text{neigh}}^{(k)}\sum_{v\in \mathcal{N}(u)} \mathbf{h}_{v}^{k-1} + \mathbf{b}^k \right). $$

## Generalized Neighborhood Aggregation

The sum over the features can lead to problems if one node has maaaany neighbors compoared to others. Therefore it can be normal to normalize based on degree:
$$ \mathbf{m}_{\mathcal{N}(u)} = \frac{\sum_{v\in\mathcal{N}(u)}\mathbf{h}_{v}}{|\mathcal{N}(u)|} \tag{average}$$

But some more success has been found with following:
$$  \mathbf{m}_{\mathcal{N}(u)} = \frac{\sum_{v\in\mathcal{N}(u)}\mathbf{h}_{v}}{|\mathcal{N}(u)||\mathcal{N}(v)|}  \tag{Symmetric}$$

However, with this there can be a loss of information!

### Graph Convolutional Network (GCN)
Popular baseline with the symmetric one above and self-loop update
$$  \mathbf{h}_{u}^{(k)} = \sigma\left( \mathbf{W}^{(k)} \sum_{v\in\mathcal{N}(u)\cup \{ u \}}\frac{\mathbf{h}_{v}}{|\mathcal{N}(u)||\mathcal{N}(v)|} \right).  $$

## Set Aggregations
We need to perform on *sets* for permutation invariance; sum is one way, but are there others?

### Set pooling
Universal set function approximator:
$$ \mathbf{m}_{\mathcal{N}(u)} = \text{MLP}_{\theta}\left( \sum_{v\in\mathcal{N}(u)}\text{MLP}_{\phi}(\mathbf{h}_{v}) \right) $$

### Janossy Pooling
Instead of permutation invariant functions as before, here we use a permutation sensitive one and average the results. Let $\pi_{i}\in \Pi$ denote permutation functions that maps sets to specific sequences $(\mathbf{h}_{v_{1}},\mathbf{h}_{v_{2}},\dots,\mathbf{h}_{|\mathcal{V}(u)|})_{\pi_{i}}$, then Janossy pooling is simply:
$$  \mathbf{m}_{\mathcal{N}(u)} = \text{MLP}_{\theta}\left( \frac{1}{|\Pi|} \sum_{\pi\in\Pi} \rho_{\phi}(\mathbf{h}_{v_{1}},\mathbf{h}_{v_{2}},\dots,\mathbf{h}_{|\mathcal{V}(u)|})_{\pi_{i}} \right). $$

### Attention
Another popular implementation is *attention*; which is weights to neighbors (not transformer attention).  So a weighted sum:
$$   \mathbf{m}_{\mathcal{N}(u)} = \sum_{v\in\mathcal{N}(u)}\alpha_{u,v}\mathbf{h}_{v}  \tag{Symmetric} $$

In the original weights are defined as
$$ \alpha_{u,v} = \frac{\exp(\mathbf{a}^\top[\mathbf{Wh}_{u}\oplus \mathbf{Wh}_{v}])}{\sum_{v'\in\mathcal{N}(u)}\exp(\mathbf{a}^\top[\mathbf{Wh}_{u}\oplus \mathbf{Wh}_{v'}])} $$

Where $\mathbf{a}$ trainable attention vector, $\mathbf{W}$ is trainable matrix and $\oplus$ is concatenation.

## Generalized Update Methods

>[!faq] Over-smoothing
>The update function can after many iterations "smooth" out the fine-grained details that differentiate nodes, and that is a problem that Generalized Update Methods can address

### Skip connections
A simple solution uses concatenation with itself to avoid this:
$$ \text{UPDATE}_{\text{concat}}(\mathbf{h}_{u},\mathbf{N}_{\mathcal{N}(u)}) = [\text{UPDATE}_{\text{base}}(\mathbf{h}_{u},\mathbf{N}_{\mathcal{N}(u)})\oplus \mathbf{h}_{u}]. $$

Another skip connection variant uses linear compositions by introducing $\mathbf{\alpha_{1},\alpha_{2}}\in[0,1]^d$ known as *gating vectors* with $\mathbf{\alpha}_{2}=\mathbf{\alpha}_1-\mathbf{1}$. (element-wise multiplication).

### Gated Updates
Uses the GRU known from RNNs!

## Edge Features & Multi-Relation GNNs
Not all graphs are simple graphs as introduced. Here are some others.

## Relational Graph NNs
This updates the aggregation function to the following by weighting based on the type:
$$ \mathbf{m}_{\mathcal{N}(u)} = \sum_{\tau\in\mathcal{R}}\sum_{v\in\mathcal{N}_{\tau}(u)} \frac{\mathbf{W}_{\tau}\mathbf{h}_{v}}{f_{n}(\mathcal{N}(u),\mathcal{N}(v))}, $$
here $f_{n}$ is a normalization function.

As can be seen, the naive approach here requires much more parameters than the one without.

# GNNs in practice

Used for 3 purposes: node classification, graph classification, or relation prediction. 

Let $\mathbf{z}_{u}\in \mathbb{R}^d$ denote the final node embedding from a GNN for node $u$, and a graph embedding as $\mathbf{z}_{\mathcal{G}}\in \mathbb{R}^d$ from a pooling function.

## Node Classification
For example paper classification where each node is a paper in relation to other via citations. Node features were language based (word vectors), and only a small subset of positive examples where given (<10%).

In these a supervised manner was approached with Softmax:
$$ \mathcal{L}=\sum_{\mathbf{u}\in \mathcal{V}_{\text{train}}}-\log(\text{Softmax}(\mathbf{z}_{u},\mathbf{y}_{u})). $$

Three types of nodes:
1) **Training Nodes**: There is the set of training nodes, $\mathcal{V}_{\text{train}}$ . These nodes are included in the GNN message passing operations, and they are also used to compute the loss.
2) **Transductive Nodes**: test nodes $\mathcal{V}_{\text{test}}$. These are unlabeled and not used for computing the loss, but is still used as part of the training because they have their embeddings calculated through message passing.
3) **Inductive Nodes**: These are completely unobserved nodes while GNN is training and are in the set $\mathcal{V}_{\text{ind}}$.


## Graph Level Implementation
Usually more efficient than the node level one and is anagolous - but memory intensive:

$$ \mathbf{H}^{(k)} = \sigma\left( \mathbf{AH}^{(k-1)}\mathbf{W}_{\text{neigh}}^{(k)}+\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}_{\text{self}} \right), $$
where $\mathbf{H}^{(k)}$ is the embedding matrix for all nodes at layer *k*.

### Sub-sambling & Mini Batching
Working with only a subset of nodes at a time, making sure we cover all nodes for a layer, can be done but will have information loss. Removing a node - even if only temporarily - will also remove its edge and therefore removal from aggregation function for that mini batch. Random selection can be really bad here. 

An idea is to sample target nodes, and then their neighbors until target batch size is hit.

## Parameter Sharing & Edge Dropout
Edge dropout is fairly straight forward and is analogous to dropout seen in Neural Networks.