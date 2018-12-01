#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "kernels.cuh"
using namespace std;
		    

int main(int argc, char ** argv){
	const long long IMAGE_SIZE = 4;
	const long long NUMB_IMAGES = 30;
	const int DIM_PROJ = 1;
	const int NUMB_PROJ = 3;
	const int ITERATIONS = 1;
	const float ALPHA = .001;
	const long long DATA_BYTES = NUMB_IMAGES*IMAGE_SIZE*sizeof(float);
	const long long NUMB_PAIRS = (NUMB_IMAGES*(NUMB_IMAGES-1))/2;
	const long long OUTPUT_BYTES = NUMB_PAIRS*IMAGE_SIZE*sizeof(float);

	// Begin importing data vector

	// Matrix of points
	float * hpoints_in = new float[IMAGE_SIZE*NUMB_IMAGES];

	// read in text file with update matrix
	ifstream inFile;
	inFile.open("data_vector_rand.txt");
	if (!inFile){
		cerr << "Unable to open file";
	exit(1);
	}

	float x;
	int i = 0;
	while (i < NUMB_IMAGES*IMAGE_SIZE){
		inFile >> x;
		hpoints_in[i] = x;
		i = i+1;
		if ((i % 10000) == 0){
			printf("%d\n",i);
		}
	}	

	inFile.close();
	
	// Matrix for random projections
	float * hrand_proj_in = new float[IMAGE_SIZE*NUMB_PROJ*DIM_PROJ];

	// read in text file with update matrix
	inFile.open("rand_proj.txt");
	if (!inFile){
		cerr << "Unable to open file";
	exit(1);
	}


	i = 0;
	while (i < IMAGE_SIZE*DIM_PROJ*NUMB_PROJ){
		inFile >> x;
		hrand_proj_in[i] = x;
		i = i+1;
		if ((i % 10000) == 0){
			printf("%d\n",i);
		}
	}	

	inFile.close();

	// Choose integer constants for algorithm
	int hsize_constants_in[7];
	// size of images
	hsize_constants_in[0] = IMAGE_SIZE;
	// number of images
	hsize_constants_in[1] = NUMB_IMAGES;
	// dimension of projection
	hsize_constants_in[2] = DIM_PROJ;
	// number of projections
	hsize_constants_in[3] = NUMB_PROJ;
	// number of iterations
	hsize_constants_in[4] = ITERATIONS;
	// number of pairs
	hsize_constants_in[5] = NUMB_PAIRS;

	// Choose float constants for algorithm
	float hfloat_constants_in[1];
	// step size
	hfloat_constants_in[0] = ALPHA;	

	// Declare auxillary arrays
	float * hsecants_out = new float[IMAGE_SIZE*NUMB_PAIRS];
	//float * hU_out = new float[IMAGE_SIZE*IMAGE_SIZE];

	//Declare handle and error objects
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	cusolverStatus_t stat_cusolve;
	cusolverDnHandle_t handle_cusolve;

	stat_cusolve = cusolverDnCreate(&handle_cusolve);
	if (stat_cusolve != CUSOLVER_STATUS_SUCCESS){
		printf("CUSOLVERDN initialization failed\n");
		cusolverDnDestroy(handle_cusolve);
		return EXIT_FAILURE;
	}

	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS initialization failed\n");
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}

	// Declare GPU memory pointers
	float * dpoints_in;
	float * dsecants_out;
	int * dsize_constants_in;
	float * drand_proj;
	float * dworst_secant_norm;
	float * dfloat_constants_in;
	float * dprojections;
	float * dsecant_norms;
	float * dwork;
	float * dprojections_reduced;
	//int lwork = 0;
	//float * dwork;
	//float * dS_out;
	//float * dU_out;
	//float * dVT_out;
	//float * dVT_trans_out;
	//int * devInfo;
	//float * rwork;
	float const alpha(1.0);
	float const beta(0.0);

	// Allocate memory on GPU for matrices and vectors
	cudaStat = cudaMalloc((void**)&dpoints_in,DATA_BYTES);
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for dpoints_in");
		return EXIT_FAILURE;
	}

	cudaStat = cudaMalloc((void**)&dsecants_out,OUTPUT_BYTES);
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for dsecant_out");
		return EXIT_FAILURE;
	}
	//cudaStat = cudaMalloc((void**)&dsecants_trans_out,OUTPUT_BYTES);
	//if (cudaStat != cudaSuccess){
	//	printf("device memory allocation failed for dsecants_trans_out");
	//	return EXIT_FAILURE;
	//}
	cudaStat = cudaMalloc((void**)&dsize_constants_in,7*sizeof(long long));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for dsize_constants_in");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dfloat_constants_in,sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for dsize_constants_in");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&drand_proj,DIM_PROJ*NUMB_PROJ*IMAGE_SIZE*sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for rand_proj");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dworst_secant_norm,ITERATIONS*NUMB_PROJ*sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for rand_proj");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dprojections,DIM_PROJ*NUMB_PROJ*NUMB_PAIRS*sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for rand_proj");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dsecant_norms,NUMB_PROJ*NUMB_PAIRS*sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for rand_proj");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dwork,DIM_PROJ*NUMB_PROJ*IMAGE_SIZE*sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for rand_proj");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dprojections_reduced,NUMB_PAIRS*IMAGE_SIZE*sizeof(float));
	if (cudaStat != cudaSuccess){
		printf("device memory allocation failed for rand_proj");
		return EXIT_FAILURE;
	}
	
	// Transfer constants
	cudaStat = cudaMemcpy(dpoints_in,hpoints_in,DATA_BYTES,cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess){
		printf("Memory transfer failed");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMemcpy(drand_proj,hrand_proj_in,DIM_PROJ*NUMB_PROJ*IMAGE_SIZE*sizeof(float),cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess){
		printf("Memory transfer failed");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMemcpy(dfloat_constants_in,hfloat_constants_in,sizeof(float),cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess){
		printf("Memory transfer failed");
		return EXIT_FAILURE;
	}
	
	// Copy points and random projects to GPU
	cudaMemcpy(dpoints_in,hpoints_in,NUMB_IMAGES*IMAGE_SIZE*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(drand_proj,hrand_proj_in,DIM_PROJ*IMAGE_SIZE*NUMB_PROJ*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dfloat_constants_in,hfloat_constants_in,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dsize_constants_in,hsize_constants_in,7*sizeof(int),cudaMemcpyHostToDevice);
	
	// Prepare to time reconstruction on GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start time
	cudaEventRecord(start);

	// Calculate secants
	calculate_secants<<<NUMB_PAIRS/29,29>>>(dsecants_out,dpoints_in,dsize_constants_in);

	//readVec<<<1,100>>>(dsecants_out);

	// Free points
	cudaFree(dpoints_in);

		
	for (int i = 0; i < ITERATIONS; i++){

		hsize_constants_in[6] = i;
		cudaMemcpy(dsize_constants_in,hsize_constants_in,7*sizeof(int),cudaMemcpyHostToDevice);
		CudaCheckError();

		// Multiply matrix of secants by matrix of projections get projections of secants
		stat = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,DIM_PROJ*NUMB_PROJ,NUMB_PAIRS,IMAGE_SIZE,&alpha,drand_proj, IMAGE_SIZE,dsecants_out,IMAGE_SIZE,&beta,dprojections,DIM_PROJ*NUMB_PROJ);
		if (stat != CUBLAS_STATUS_SUCCESS){
			printf("Secant projection failed\n");	
			cudaFree(dsecants_out);
			cudaFree(dsize_constants_in);
			cudaFree(dprojections);
			cudaFree(drand_proj);
			cudaFree(dfloat_constants_in);
		}

		//readVec<<<1,100>>>(dprojections);

		// Multiply matrix of projectons by matrix of projections get projections of secants
		stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,IMAGE_SIZE,NUMB_PAIRS,DIM_PROJ*NUMB_PROJ,&alpha,drand_proj, IMAGE_SIZE,dprojections,DIM_PROJ*NUMB_PROJ,&beta,dprojections_reduced,IMAGE_SIZE);
		if (stat != CUBLAS_STATUS_SUCCESS){
			printf("Secant projection failed\n");	
			cudaFree(dsecants_out);
			cudaFree(dsize_constants_in);
			cudaFree(dprojections);
			cudaFree(drand_proj);
			cudaFree(dfloat_constants_in);
		}	
		
		// Run kernel for finding worst secant and adjusting projection
		calculate_projection_norms<<<NUMB_PAIRS/29,29>>>(dprojections,dsecant_norms,dsize_constants_in);		CudaCheckError();	
		
		//readVec<<<1,100>>>(dsecant_norms);

		// Run kernel to update projections

		projection_refinement<<<1,NUMB_PROJ>>>(dworst_secant_norm,dprojections,dsecant_norms,drand_proj, dsize_constants_in,dfloat_constants_in,dwork,dprojections_reduced, dsecants_out);
		CudaCheckError();

	}


	readVec<<<1,IMAGE_SIZE*NUMB_PROJ*DIM_PROJ>>>(drand_proj);	
	CudaCheckError();

		// Stop timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds,start,stop);
	printf("Elapsed GPU time in milliseconds is %f\n",milliseconds);

	float * hworst_secant_norms = new float[ITERATIONS*NUMB_PROJ];
	cudaMemcpy(hrand_proj_in,drand_proj,IMAGE_SIZE*DIM_PROJ*NUMB_PROJ*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(hworst_secant_norms,dworst_secant_norm,ITERATIONS*NUMB_PROJ*sizeof(float),cudaMemcpyDeviceToHost);

	// print some secants
	for (int i = 0; i < 10; i++){
		printf("%dth element is %f\n",i,hrand_proj_in[i]);
	}
	// print some worst projections
	for (int i = 0; i < 10; i++){
		printf("%dth element is %f\n",i,hworst_secant_norms[i]);
	}
	//for (int i = 1; i < 10; i++){
	//	printf("%lldth SVD element is %f\n",i,hU_out[i]);
	//}

	// open file to store output
	ofstream myFile;
	myFile.open("secants.txt");

	for (int j = 0; j < NUMB_PAIRS*IMAGE_SIZE; j++){
		myFile << hsecants_out[j] << "\n";
		if ((j % 1000000) == 0){
			printf("Transfering back element %d\n",j);
		}
	}

	myFile.close();
	
	// Delete arrays on host
	//delete hpoints_in;
	delete hsecants_out;
	//delete hU_out;
	delete hrand_proj_in;

	return 0;

}









