# The secant avoidance projection algorithm

The secant avoidance projection (SAP) algorithm is a secant-based dimensionality reduction algorithm. 
Very roughly it searches for projections that preserve the spatial relations among points in a data set.
This is equivalent to preserving the secant set of the data set.

More formally, the optimization problem that this algorithm solves is

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{argmax}_{P&space;\in&space;\text{Proj}(n,m)}&space;\min_{s&space;\in&space;S}&space;||P^Ts||_{\ell_2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{argmax}_{P&space;\in&space;\text{Proj}(n,m)}&space;\min_{s&space;\in&space;S}&space;||P^Ts||_{\ell_2}" title="\text{argmax}_{P \in \text{Proj}(n,m)} \min_{s \in S} ||P^Ts||_{\ell_2}" /></a>

This repository contains CUDA code for the SAP algorithm.



