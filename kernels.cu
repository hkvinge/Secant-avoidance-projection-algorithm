#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "kernels.cuh"
using namespace std;
		    

// Read entries of vector
__global__ void readVec(float * dvector_in){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < 500){
		printf("The entry at %d is %f\n",idx,dvector_in[idx]);
	}
}

// Read entries of vector (int)
__global__ void readVecInt(int * dvector_in){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < 500){
		printf("The entry at %d is %i\n",idx,dvector_in[idx]);
	}
}

// Calculate the secants for a collection of vectors stored as columns in a matrix, then normalize
__global__ void calculate_secants(float * dsecants_out, float * dpoints_in, int * dsize_constants_in){


	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int n = dsize_constants_in[1];
	int image_size = dsize_constants_in[0];
	int i = idx % (n-1);
	int j = (idx - i)/(n-1);
	int pair1;
	int pair2;
	if (i >= j){
		pair1 = i+1;
		pair2 = j;
	}else{
		pair1 = n-i-1;
		pair2 = n-j-1;
	}
	for (int p = 0; p < image_size; p++){
		dsecants_out[idx*image_size + p] = dpoints_in[image_size*pair1 + p] - dpoints_in[image_size*pair2 + p];
	}
	float norm = 0;
	for (int p = 0; p < image_size; p++){
		norm = norm + powf(dsecants_out[idx*image_size + p],2);
	}
	norm = sqrtf(norm);
	if (norm != 0.0){
		for (int p = 0; p < image_size; p++){
			dsecants_out[idx*image_size + p] = (1/norm)*dsecants_out[idx*image_size + p];
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

// Take a given matrix and turn it into an identity matrix
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









