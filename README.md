# The secant-avoidance projection algorithm

The secant avoidance projection (SAP) algorithm is a secant-based dimensionality reduction algorithm. 
Very roughly it searches for projections that preserve the spatial relations among points in a data set.
This is equivalent to preserving the secant set of the data set. 

The code found in this repository is written in C++/CUDA to run SAP on Nvidia GPU's. 

# What is in this repository?

This repository contains the following files:

* SAP.cu: Contains the **SAP** function as well as a little CUDA error correcting code. The SAP function transfers data (both the actual data points and an initial projection) to the GPU where the algorithm is run by making library calls to CUBLAS and CUSOLVER and also calling certain custom kernels from kernels.cu.

* kernels.cu: Contains the custom kernels called by the SAP function. Most of these are fairly elementary. I expect that some of these could actually be replaced by calls to the CUBLAS library. This may be updated in the future. 

* kernels.cuh, SAP.cuh: CUDA header files.

* example_call.cu: Contains an example call of the SAP function.

* initial_projection: Currently, example_call.cu loads an initial projection from the text file initial_projection.txt. This initial projection is stored in a text file with one float per line. Despite being essentially 1-dimensional, this array can be thought of as a major stored in column

* example_data_points.txt: This text file contains some sample data points that example_call.cu loads and runs SAP on. The text file contains one float per line. This (effectively 1-dimensional array) should be treated as a matrix in column major format. That is, the 1-dimensional array should be read in as many stacked columns. Data points correspond to these columns. The dimension of the data points (in this case 10) is the first entry of the text file. The second entry is the number of points (in this case 256). Thus the data in this file can be thought of as a (10 x 256) matrix (after the first two entries have been removed). 

# Mathematical details

More formally, the optimization problem that this algorithm solves is

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{argmax}_{P&space;\in&space;\text{Proj}(n,m)}&space;\min_{s&space;\in&space;S}&space;||P^Ts||_{\ell_2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{argmax}_{P&space;\in&space;\text{Proj}(n,m)}&space;\min_{s&space;\in&space;S}&space;||P^Ts||_{\ell_2}" title="\text{argmax}_{P \in \text{Proj}(n,m)} \min_{s \in S} ||P^Ts||_{\ell_2}" /></a>

This repository contains CUDA code for the SAP algorithm.



