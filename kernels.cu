#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "kernels.cuh"
using namespace std;
		    
__global__ void readVec(float * dvector){
	/**
	Read some number (specified by the number of threads at execution)
	of entries from an array of floats on the device. Useful to debugging.

	@param dvector The name of the array of floats on the device that you
	wish to read entries from
	*/
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("The entry at %d is %f\n",idx,dvector[idx]);
}

__global__ void calculate_secants(float * dsecants_out, float * dpoints_in, int * dsize_constants_in){
	/** 
	Read in a set of data points given as a 1-dimensional float array and output 
	1-dimensional float array with all normalized secants between these points.
	Assume that there are N points which live in dimension D. Then there will
	be S := N(N-1)/2 secants. In order to bijectively assign a thread id 
	(we assume that the kernel is called so that the number of threads matches
	the number of secants S), we use the map which takes 
	an integer k from {1,2,...,S} and assigns it to (i,j) such that
	k = Nj + i. If 

	@param dpoints_in The array of floats giving the collection of data points
	in question. Note that we stack all points so that the length of this
	array should be (number of points)*(dimenion of space where points live).

	@param dsecants_out The array of floats giving the collection of normalized
	secants between all data points in dpoints_in. Note that we stack all 
	secants so that if we originally have N points, the length of dsecants_out is 
	N(N-1)/2 * (dimension of space where points live).

	@param dsize_constants_in An array of integers holding important size 
	constants. 
		dsize_constants_in[0] = dimension where points live
		dsize_constants_in[1] = number of points.

	*/
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	// The number of data points
	int n = dsize_constants_in[1];
	// Dimension where data lives
	int data_dim = dsize_constants_in[0];
	// Coordinates for assigning a thread to a pair of data points
	int i = idx % (n-1);
	int j = (idx - i)/(n-1);
	// Index for first data point in pair
	int pair1;
	// Index for second data point in pair
	int pair2;

	if (i >= j){
		pair1 = i+1;
		pair2 = j;
	}else{
		pair1 = n-i-1;
		pair2 = n-j-1;
	}
	for (int p = 0; p < data_dim; p++){
		dsecants_out[idx*data_dim + p] = dpoints_in[data_dim*pair1 + p] - dpoints_in[data_dim*pair2 + p];
	}
	float norm = 0;
	// Calculate the norm of the secant in order to normalize it
	for (int p = 0; p < data_dim; p++){
		norm = norm + powf(dsecants_out[idx*data_dim + p],2);
	}
	norm = sqrtf(norm);
	// If the length is greater than zero, normalize the secant
	// and assign it to the secant array.
	if (norm != 0.0){
		for (int p = 0; p < data_dim; p++){
			dsecants_out[idx*data_dim + p] = (1/norm)*dsecants_out[idx*data_dim + p];
		}
	}
}

// Take a matrix and return a vector whose entries are the l2 norms of the matrix
__global__ void calculate_col_norms(float * dprojected_secants, float * dsecant_norms, int * dsize_constants_in){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int proj_dim = dsize_constants_in[2];
	float sum = 0;
	for (int i = 0; i < proj_dim; i++){
		sum = sum + powf(dprojected_secants[idx*proj_dim + i],2);
	}
	dsecant_norms[idx] = sqrtf(sum);
}

// Switch two columns in the matrix
__global__ void switch_columns(float * matrix, int * dcolumn_switch_constants){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float temp;
	temp = matrix[(dcolumn_switch_constants[1])*dcolumn_switch_constants[0] + idx];
	matrix[(dcolumn_switch_constants[1])*dcolumn_switch_constants[0] + idx] =  matrix[(dcolumn_switch_constants[2])*dcolumn_switch_constants[0] + idx];
	matrix[(dcolumn_switch_constants[2])*dcolumn_switch_constants[0] + idx] = temp;
}

// Shift vector in first column of projection
__global__ void shift_first_column(float * matrix, float * projection,float * shortest_secant, float * dalgo_constant){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float alpha = dalgo_constant[0];
	matrix[idx] = (1.0-alpha)*projection[idx] + alpha*(shortest_secant[idx] - projection[idx]);
}

// Normalize first column
__global__ void normalize_first_column(float * dproj, int * dsize_constants_in){
	float sum = 0;
	int column_height = dsize_constants_in[0];
	for (int i = 0; i < column_height; i++){
		sum = sum + powf(dproj[i],2);
	}
	sum = sqrtf(sum);
	if (sum != 0.0){
		for (int i = 0; i < column_height; i++){
			dproj[i] = (1/sum)*dproj[i];
		}
	}	
}

// Take a given matrix and turn it into an identity matrix of given dimension
__global__ void make_identity(float * didentity_matrix, int * dsize_constants_in){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int col = idx/dsize_constants_in[2];
	int row = idx - col*dsize_constants_in[2];
	if (row == col){
		didentity_matrix[idx] = 1.0;
	}else{
		didentity_matrix[idx] = 0.0;
	}
}
