# Manifold-Learning-and-Graph-Kernels
Third assignment for A.I. course, Prof. Torsello, Ca' Foscari University of Venice, A.Y. 2018/2019

## Assignment
<p>Read <a href="http://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf">this article</a> presenting a way to improve the disciminative power of graph kernels.</p>
<p>Choose one <a href="http://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf">graph kernel</a> among
<ul>
	<li>Shortest-path Kernel</li>
	<li>Graphlet Kernel</li>
	<li>Random Walk Kernel</li>
	<li>Weisfeiler-Lehman Kernel</li>
</ul>
</p>
<p>Choose one manifold learning technique among
<ul>
	<li>Isomap</li>
	<li>Diffusion Maps</li>
	<li>Laplacian Eigenmaps</li>
	<li>Local Linear Embedding</li>
</ul>
</p>
<p>
Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:
<ul>
	<li><a href="http://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat">PPI</a></li>
	<li><a href="http://www.dsi.unive.it/~atorsell/AI/graph/SHOCK.mat">Shock</a></li>
</ul>
</p>
<p><b>Note:</b> the datasets are contained in Matlab files. The variable G contains a vector of cells, one per graph. 
The entry am of each cell is the adjacency matrix of the graph.
The variable labels, contains the class-labels of each graph.	
</p>
