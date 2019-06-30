---
title: "Artificial Intelligence - Knowledge Representation and Planning - Assignment 3"
author: "Lorenzo Soligo - 875566"
date: "Academic Year 2018-2019"
subparagraph: yes
numbersections: true
documentclass: article
# fontfamily: mathpple
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

* [PPI](http://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat)
* [Shock](http://www.dsi.unive.it/~atorsell/AI/graph/Shock.mat)

**Note**: the datasets are contained in Matlab files. The variable G contains a vector of cells, one per graph. 
The entry am of each cell is the adjacency matrix of the graph.
The variable labels, contains the class-labels of each graph.	


# Graph Kernels
## Graph Comparison Problem
Given two graphs $G$ and $G'$ from the space of graphs $\mathcal{G}$, the problem of graph comparison is to find a mapping $$s:\mathcal{G}\times \mathcal{G}\rightarrow \mathbb{R}$$ such that $s(G, G')$ quantifies the similarity (or dissimilarity) of $G$ and $G'$. 

## Graph isomorphism
Given two graphs $G_1$ and $G_2$, find a mapping $f$ of the vertices of $G_1$ to the vertices of $G_2$ such that $G_1$ and $G_2$ are identical, i.e. $(x,y)$ is an edge of $G_1$ iff $(f(x), f(y))$ is an edge of $G_2$. Then $f$ is an isomorphism, and $G_1$ and $G_2$ are said to be isomorphic.
\newline
At the moment we do not know a polynomial time algorithm for graph isomorphism, but we also do not know whether the problem is NP-complete.
\newline
On the other hand, we know that subgraph isomorphism is NP-complete. Subgraph isomorphism checks whether there is a subset of edges and vertices of $G_1$ that is isomorphic to a smaller graph $G_2$.

## Graph edit distances 
The idea is to count the number 





# Resources
* Professor's slides
* http://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf
* http://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf
* https://www.ethz.ch/content/dam/ethz/special-interest/bsse/borgwardt-lab/documents/slides/CA10_GraphKernels_intro.pdf
* https://scikit-learn.org/stable/modules/manifold.html
* 