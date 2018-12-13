/*! \file
*  \brief Custom GPU kernels used in the SAP algorithm
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "kernels.cuh"
using namespace std;


__global__ void readVec(float * d_vector){
	/**
	Reads some number (specified by the number of threads at execution)
	of entries from an array of floats on the device. Useful for debugging.

	@param d_vector The name of the array of floats on the device
	which the kernel prints entries from. 
	*/
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("The entry at %d is %f\n",idx,dvector_in[idx]);
}

__global__ void readVecInt(int * d_vector){
	/**
	Reads some number (specified by the number of threads at execution)
	of entries from an array of floats on the device. Useful for debugging.

	@param d_vector The name of the array of floats on the device
	which the kernel prints entries from. 
	*/
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("The entry at %d is %i\n",idx,d_vector[idx]);
}

// Calculate the secants for a collection of vectors stored as columns in a matrix, then normalize
__global__ void calculate_secants(float * d_secants, float * dpoints_in, int * d_int_constants){
	/** 
	Calculates the normalized secant set for a set of points.

	@param d_secants The secant set for d_points. Secants are stored as column
	vectors so the dimension of this matrix is (dim. of input data x number of
	secants). This matrix is the output of this kernel.
	@param d_points The input points stored as column vectors. The dimension
	of this matrix is (dim. of input data x number of points in data set)..
	@param d_int_constants An integer array which holds the input dimension 
	(the second entry in the array) and number of points (the second 
	entry in the array)

	*/

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	// Number of points
	int n = d_int_constants[1];
	// Input dimension
	int input_dim = d_int_constants[0];
	// Parameters used to pair points to calculate secants
	int i = idx % (n-1);
	int j = (idx - i)/(n-1);
	// Number of points for given pair
	int pair1;
	int pair2;
	if (i >= j){
		pair1 = i+1;
		pair2 = j;
	}else{
		pair1 = n-i-1;
		pair2 = n-j-1;
	}
	// For loop calculates secant coordinate by coordinate
	for (int p = 0; p < input_dim; p++){
		d_secants[idx*input_dim + p] = dpoints_in[input_dim*pair1 + p] - dpoints_in[input_dim*pair2 + p];
	}
	// Variable to store the norm of the secant
	float norm = 0;
	// Iterate through entries of the secant to calculate its norm
	for (int p = 0; p < input_dim; p++){
		norm = norm + powf(d_secants[idx*input_dim + p],2);
	}
	norm = sqrtf(norm);
	// As long as the norm is not zero, normalize the secant
	if (norm != 0.0){
		for (int p = 0; p < input_dim; p++){
			d_secants[idx*input_dim + p] = (1/norm)*d_secants[idx*input_dim + p];
		}
	}
}

// Take a matrix and return a vector whose entries are the l2 norms of the matrix
__global__ void calculate_col_norms(float * d_matrix, float * d_col_norms, int * d_int_constants){
	/** 
	Takes a matrix and returns a row vector whose entries are the L2 norms of each
	entry in the matrix.

	@param d_matrix The matrix whose column norms are being calculated 
	@param d_col_norms The row vector which stores the calculated column norms
	@param d_int_constants An integer array that stores the dimensions of the matrix. The third
	entry is the length of each column.

	*/

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	// Get the length of columns
	int proj_dim = d_int_constants[2];
	// Summation variable to store column norms
	float sum = 0;
	// Iterate down the columns, summing the squares of the entries
	for (int i = 0; i < proj_dim; i++){
		sum = sum + powf(d_matrix[idx*proj_dim + i],2);
	}
	// Take square root of resulting sum
	d_col_norms[idx] = sqrtf(sum);
}

__global__ void switch_columns(float * d_matrix, int * d_column_switch_indices){
	/** 
	Takes a matrix and switches two specified columns.

	@param d_matrix The matrix whose column norms are being calculated 
	@param d_col_norms The row vector which stores the calculated column norms
	@param d_int_constants An integer array that stores the dimensions of 
	the matrix. The first entry is the number of rows of the matrix. The 
	second and the third are the indices of the columns to be switched.

	*/

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	// Initialize float to hold values from was column while the other is 
	// copied over
	float temp;
	// Number of rows
	int rows = d_column_switch_indices[0];
	// Index of first column to be swapped
	int index1 = d_column_switch_indices[1];
	// Index of second column to be swapped
	int index2 = d_column_switch_indices[2];
	// Save first column as temp
	temp = matrix[index1*rows + idx];
	// Copy second column into first index position
	matrix[index1*rows + idx] =  matrix[index2*rows + idx];
	// Copy original first column into second index position
	matrix[index2*rows + idx] = temp;
}

// Shift vector in first column of projection
__global__ void shift_first_column(float * matrix, float * projection,float * shortest_secant, float * dalgo_constant){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float alpha = dalgo_constant[0];
	matrix[idx] = (1.0-alpha)*projection[idx] + alpha*(shortest_secant[idx] - projection[idx]);
}

// Normalize first column
__global__ void normalize_first_column(float * dproj, int * d_int_constants){
	float sum = 0;
	int column_height = d_int_constants[0];
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

// Take a given matrix and turn it into an identity matrix
__global__ void make_identity(float * didentity_matrix, int * d_int_constants){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int col = idx/d_int_constants[2];
	int row = idx - col*d_int_constants[2];
	if (row == col){
		didentity_matrix[idx] = 1.0;
	}else{
		didentity_matrix[idx] = 0.0;
	}
}









