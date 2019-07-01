---
title: "Artificial Intelligence - Knowledge Representation and Planning - Assignment 3"
author: "Lorenzo Soligo - 875566"
date: "Academic Year 2018-2019"
subparagraph: yes
numbersections: true
documentclass: article
urlcolor: blue
geometry: [left=3.7cm, top=3.2cm, right=3.7cm, bottom=3.2cm]
---

\clearpage

# Requirements 
Read [this article](http://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf) presenting a way to improve the disciminative power of graph kernels.
Choose one [graph kernel](http://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf) among

* Shortest-path Kernel
* Graphlet Kernel
* Random Walk Kernel
* Weisfeiler-Lehman Kernel


Choose one manifold learning technique among

* Isomap
* Diffusion Maps
* Laplacian Eigenmaps
* Local Linear Embedding


Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:

* [PPI](http://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat): this is a Protein-Protein Interaction dataset. Here proteins (nodes) are connected by an edge in the graph if in actuality they are so physically or functionally.
* [Shock](http://www.dsi.unive.it/~atorsell/AI/graph/Shock.mat): representing 2D shapes.

**Note**: the datasets are contained in Matlab files. The variable G contains a vector of cells, one per graph. 
The entry am of each cell is the adjacency matrix of the graph.
The variable labels, contains the class-labels of each graph.	


# Background
## Kernel functions
A positive-definite kernel is a generalization of a  positive-definite function or a positive-definite matrix.

Let $\mathcal{X}$ be a nonempty set. A *symmetric* function $K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ is a positive-definite kernel on $\mathcal{X}$ if 
$$ 
\sum_{i=1}^n\sum_{j=1}^nc_ic_jK(x_i,x_j)\ge 0 \, \, \ \ \ \ \forall n \in \mathbb{N},\ \  \forall x_1, ..., x_n \in \mathcal{X},\ \ \forall c_1, ..., c_n \in \mathbb{R}
$$
Notice that positive definite kernels require $c_i=0 \ \forall i$, while positive semi-definite kernels do not impose this condition. This relates to the spectrum of a finite matrix constructed by pairwise evaluation $\textbf{K}_{ij} = K(x_i, x_j)$: in the former case we have only positive eigenvalues; in the latter we have non-negative eigenvalues.

### The kernel trick
Kernel methods (namely SVMs and many more) exploit kernel functions to work on high-dimensional, implicit feature spaces without having to compute the coordinates of the data in that space. This is achieved by performing inner products between the images of all pairs of data in the feature space. This operation is called *kernel trick*. It is extremely useful in the case the dataset is not linearly separable, but can be easily separated by an hyperplane in a higher-dimensional space.

Formally, a kernel maps two objects $x$ and $x'$ via a mapping $\phi$ into the feature space $\mathcal{H}$, measuring their similarity in $\mathcal{H}$ as $\big \langle\phi(x),\phi(x')\big \rangle$. The kernel trick is nothing but computing the inner product in $\mathcal{H}$ as kernel in the input space: $k(x,x')=\big \langle\phi(x),\phi(x')\big \rangle$.

## Graph Comparison Problem
Given two graphs $G$ and $G'$ from the space of graphs $\mathcal{G}$, the problem of graph comparison is to find a mapping $$s:\mathcal{G}\times \mathcal{G}\rightarrow \mathbb{R}$$ such that $s(G, G')$ quantifies the similarity (or dissimilarity) of $G$ and $G'$. 

## Graph isomorphism
Given two graphs $G_1$ and $G_2$, find a mapping $f$ of the vertices of $G_1$ to the vertices of $G_2$ such that $G_1$ and $G_2$ are identical, i.e. $(x,y)$ is an edge of $G_1$ iff $(f(x), f(y))$ is an edge of $G_2$. Then $f$ is an isomorphism, and $G_1$ and $G_2$ are said to be isomorphic.
\newline
At the moment we do not know a polynomial time algorithm for graph isomorphism, but we also do not know whether the problem is NP-complete.
\newline
On the other hand, we know that subgraph isomorphism is NP-complete. Subgraph isomorphism checks whether there is a subset of edges and vertices of $G_1$ that is isomorphic to a smaller graph $G_2$.

### Graph edit distances 
The idea is to count the number of operations that is necessary to transform $G_1$ into $G_2$, assigning different costs to the different types of operations (e.g. edge/node insertion/deletion, modification of labels, ...). This allows us to capture (partial) similarities between graphs, but contains a check for subgraph isomorphism (which is NP-complete) as an intermediate step.

### Topological descriptors
The idea here is to map each graph to a feature vector and then using distances and metrics on vectors for learning on graphs. In this case the clear advantage is that known, efficient tools for feature vectors can be reused, but the feature vector transformation either leads to a loss of topological information or still includes subgraph isomorphism as one step.


# Graph kernels
## Introduction
From the background, we have understood that computing whether two graphs are isomorphic is usually expensive, often becoming infeasible for "big" graphs. Therefore it would be great to have a polynomial time similarity measure for graphs. Graph kernels allow us to compare substructures of graphs that are computable in polynomial time. We want a graph kernel to be expressive, efficient to compute, positive definite and applicable to a wide range of graphs.

## Representation of graphs
Graphs are usually represented using adjacency lists/matrices. However, standard pattern recognition techniques require data to be represented in vectorial form. This is quite a tough operation for graphs. First of all, the nodes in a graph are not ordered, therefore a reference structure must be established as a prerequisite. Second, even though the vectors could be encoded as vectors, their length would be variable and they would therefore belong to different spaces.


### The kernel trick, again
The kernel trick has the advantage of shifting the problem from a vectorial representation -now implicit- to a similarity representation, allowing standard learning techniques to be applied to data for which a vectorial representation is hard to achieve.


## Definition and problems
### What is a graph kernel?
First of all, we need to define R-convolution kernels. These kernels compare decompositions of two structured objects. Most R-convolution kernels simply count the number of isomorphic substructures in the two compared graphs and differ mainly by the type of substructures used in the deconvolution and the algorithms used to count them efficiently. 
$$k_{convolution}(x, x') = \sum_{(x_d,x)\in\mathbb{R}}\sum_{(x_d',x')\in\mathbb{R}}k_{parts}(x_d,x_d')$$
Graph kernels are nothing but convolution kernels on pairs of graphs. A new decomposition relation $R$ results in a new graph kernel (???). A graph kernel makes the whole family of kernel methods applicable to graphs.

Formally, once we define a positive semi-definite kernel $k:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ on a set $X$, there exists a map $\phi : X \rightarrow \mathcal{H}$ into a Hilbert space $\mathcal{H}$ such that $k(x,y)=\phi(x)^T\phi(y)\ \ \forall x,y\in X$. Also, the distance between $\phi(x)$ and $\phi(y)$ can be computed as 
$$
||\phi(x), \phi(y)||^2 = \phi(x)^T\phi(x) + \phi(y)^T\phi(y) -2\phi(x)^T\phi(y)
$$

### Link to graph isomorphism (hardness result)
One of the main problems with the aforementioned approach is that given the high degree of information that graphs express, the task of defining complete kernels (i.e. $\phi$ is injective) is proved to be as hard as solving the graph isomorphism problem. 

In particular, let $k(G,G')=\big \langle\phi(G),\phi(G')\big \rangle$ be a graph kernel. Let $\phi$ be injective. Then, computing any complete graph kernel is at least as hard as deciding whether two graphs are isomorphic.

In fact, since $\phi$ is injective, we have

\begin{align*}
&\sqrt{k(G, G)-2k(G,G')+k(G',G')} \\
&=\sqrt{\big \langle \phi(G)-\phi(G'), \phi(G)-\phi(G')   \big \rangle} \\
&=||\phi(G)-\phi(G')|| = 0
\end{align*}
iff $G$ is isomorphic to $G'$.

### Complexity and horseshoe effect
Many graph kernels are very effective in generating implicit embeddings, but there is no guarantee that the data in the Hilbert space will show a better class separation. This happens because of the complexity of the structural embedding problem and the limits for efficient kernel computation. For example, data tends to cluster tightly along a curve that wraps around the embedding space due to the consistend underestimation of the geodesic distances on the manifold, placing data onto a highly non-linear manifold in the Hilbert space. As a matter of fact, this *horseshoe* is the intersection between the manifold and the plane used to visualise the data. It might be caused by kernel normalisation, that projects data points from the Hilbert space to the unit sphere possibly creating an artificial curvature of the space that either generates or exaggerates the horseshoe effect.

### Locality
Generally the non-linearity of the mapping is used to improve local class separability, while a large curvature might fold the manifold reducing long range separability. The impact of the locality of distance information on the performance of the kernel thus becomes a key point to be studied: we will use some manifold learning techniques to embed the graphs onto a low-dimensional vectorial space, trying to unfold the embedding manifold and increase class separation.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{img/embedding.png}
\caption{Example of reduced linear separability due to high curvature of the embedding. Introducing a non-linear mapping to a low-curvature manifold makes the data linearly separable. Mapping to high global curvature manifold results in low linear separability of the data. The higher the curvature the less separable the data is.}
\end{figure}

Also, many kernels proposed in the literature neglet locational information for the substructures in a graph, and cannot therefore establish reliable structural correspondences between the substructures in a pair of graphs, lowering the precision of the similarity measure.




# Manifold Learning
## Introduction and motivations
Manifold learning is an approach to non-linear dimensionality reduction. Algorithms for this task are based on the idea that the dimensionality of many data sets is only artificially high and the data actually resides in a low-dimensional manifold embedded in the high-dimensional feature space. Also, the manifold may fold or wrap in the feature space so much that the natural feature-space parametrization does not capture the underlying structure of the problem.
Manifold learning algorithms attempt to uncover a non-linear parametrization for the data manifold in order to find a low-dimensional representation of the data that effectively unfolds the manifold and reveals the underlying data structure.

High-dimensional datasets can be very difficult to visualize. In order to visualize the structure of a dataset, the dimension must be reduced in some way.
The simplest way to accomplish this dimensionality reduction is by taking a random projection of the data. Though this allows some degree of visualization of the data structure, the randomness of the choice leaves much to be desired. In a random projection, it is likely that the more interesting structure within the data will be lost.

To address this concern, a number of supervised and unsupervised linear dimensionality reduction frameworks have been designed, such as Principal Component Analysis (PCA) and many others. These algorithms define specific rubrics to choose an "interesting" linear projection of the data. These methods can be powerful, but often miss important non-linear structure in the data.

\begin{figure}[H]
  \centering
  \begin{minipage}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{img/sphx_glr_plot_lle_digits_001.png}
  \end{minipage}
  \hfill
  \begin{minipage}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{img/sphx_glr_plot_lle_digits_002.png}
  \end{minipage}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.45\textwidth]{img/sphx_glr_plot_lle_digits_004.png}
\caption{The representation drastically improves using dimensionality reduction techniques}
\end{figure}

## Defining manifold learning
Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data. Though supervised variants exist, the typical manifold learning problem is unsupervised: it learns the high-dimensional structure of the data from the data itself, without the use of predetermined classifications.
Intuitively, the "curvier" is the considered manifold, the denser the data must be.

Now we will define the two manifold learning algorithms used in this assignment: we will see a global approach (Isomap) and a local one (LLE).


## Isomap
### Description
Isomap (short for **iso**metric feature **map**ping) seeks a low-dimensional representation of the data which maintains geodesic (namely, the shortest path between two points on a surface/manifold) distances between all points. In this sense, it is a direct generalization of Multidimensional Scaling (MDS). The geodesic distances are approximated as the length of the minimal path on a neighborhood graph. In Isomap, the long-range distances become more important than the local structure, and this makes it quite sensitive to noise: depending on the topology of the neighborhood graph, Isomap suffers shortcutting and other distortions. Isomap is computational expensive due to the heavy work on matrices to be done.

### Steps
Isomap consists of three stages:

1. Nearest-neighbor search
2. Shortest-path graph search
3. Partial eigenvalue decomposition


## Locally Linear Embedding (LLE)
Locally linear embedding (LLE) seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods. It can be thought of as a series of local Principal Component Analyses which are globally compared to find the best non-linear embedding. Here the manifold is seen as a collection of overlapping coordinate patches: if the neighborhoods are small enough and the manifold is smooth enough, the local geometry of the patches can be considered approximately linear. 
Since it focuses on preserving distances locally, LLE can distort the global structure of the data. The idea, in fact, is to characterize the local geometry of each neighborhood as a linear function and to find a mapping to a lower dimensional Euclidean space that preserves the linear relationship between a point and its neighbors.


### Steps
The standard LLE algorithm consists of three stages:

1. Nearest-neighbor search
2. Weight matrix construction
3. Partial eigenvalue decomposition

### Modified Locally Linear Embeddng
One well-known issue with LLE is the regularization problem. When the number of neighbors is greater than the number of input dimensions, the matrix defining each local neighborhood is rank-deficient. Standard LLE addresses this problem by applying an arbitrary regularization parameter $r$, which is chosen relative to the trace of the local weight matrix. It can be proved that for $r \rightarrow 0$ the solution converges to the desired embedding, but there is no guarantee that the optimal solution will be found for $r>0$. This results in a distortion of the underlying geometry of the manifold.

One method to address the regularization problem is to use multiple weight vectors in each neighborhood. This is the essence of modified locally linear embedding (MLLE). 

The steps taken are the same as standard LLE, but the weight matrix construction takes more time because we need to construct the weight matrix from multiple weights. In practice, however, this increase in the cost is negligible with respect to the cost of steps 1 and 3.


# Graph Kernels and Manifold Learning
## Challenges
As previously said, applying multidimensional scaling to the distances in the implicit Hilbert space obtained from R-convolution type graph kernels often results in the horseshoe effect, meaning that data is distributed tightly on a highly curved line or manifold. This comes from a consistent underestimation of the long-range distances consistent with the properties of these kernels.

R-convolution kernels typically count the number of isomorphic substructures in the decomposition of the two graphs, not considering locational information for the substructures in a graph - i.e., the similarity of the substructures are not related to the relative position in the graphs. When graphs are very dissimilar, many similar small substructures can appear simply because of the statistics of random graphs, and the smaller and simpler the substructures are in the decomposition, the more likely it is to find them in many locations of the two structures. In other words, the smaller is the considered sample, the higher is the probability of finding similarities because of random fluctuations. Notice that it is the proportion of correct matches with respect to the total possible correspondences of the same size that decreases as the size increases.

The lack of a locality condition and the consequent summation over the entire structure amplifies the effects of these random similaritis, resulting in a lower bound on the kernel value that is a function only of the random graph statistics. This leads to a consistent reduction in the estimated distances for dissimilar graphs, adding a strong curvature to the embedding manifold -which can fold on itself- and increasing the effective dimensionality of the embedding.

# Experiments
## Kernel 1
## Kernel 2

# Conclusions




# Resources
* Professor's slides
* http://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf
* http://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf
* https://en.wikipedia.org/wiki/Kernel_method
* https://en.wikipedia.org/wiki/Positive-definite_kernel
* https://www.ethz.ch/content/dam/ethz/special-interest/bsse/borgwardt-lab/documents/slides/CA10_GraphKernels_intro.pdf
* https://scikit-learn.org/stable/modules/manifold.html
* https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html 