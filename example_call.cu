#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include "SAP.cuh"
using namespace std;
		    

int main(int argc, char ** argv){

	// Choose key parameters of SAP
	int iterations = 50;
	float step_size = .05;

	// Initialize parameters to characterize data
	int input_dim;
	int numb_points;
	int proj_dim;

	// Begin importing data vector

	// read in text file with update matrix
	ifstream inFile;
	inFile.open("example_data_points.txt");
	if (!inFile){
		cerr << "Unable to open file";
	exit(1);
	}

	inFile >> input_dim;
	inFile >> numb_points;

	// Matrix of points
	float * h_points_in = new float[input_dim*numb_points];
	
	float x;
	int i = 0;
	while (i < input_dim*numb_points){
		inFile >> x;
		h_points_in[i] = x;
		i = i+1;
		if ((i % 10000) == 0){
			printf("%d\n",i);
		}
	}	

	inFile.close();
	
	// read in text file with update matrix
	inFile.open("initial_projection.txt");
	if (!inFile){
		cerr << "Unable to open initial projection file";
	exit(1);
	}

	inFile >> proj_dim;

	// Matrix for random projections
	float * h_proj = new float[input_dim*proj_dim];

	i = 0;
	while (i < proj_dim*input_dim){
		inFile >> x;
		h_proj[i] = x;
		i = i+1;
		if ((i % 10000) == 0){
			printf("%d\n",i);
		}
	}	

	inFile.close();

	// Create file to store the smallest secant norms
	float h_smallest_secant_norms[iterations];

	SAP(input_dim, numb_points, h_points_in, proj_dim, h_proj, h_smallest_secant_norms, iterations, step_size);

	// open file to store output
	ofstream myFile;
	myFile.open("output_projection.txt");

	for (int j = 0; j < proj_dim; j++){
		myFile << h_proj[j] << "\n";
	}

	myFile.close();
	
	// Delete arrays on host
	delete h_points_in;
	delete h_proj;

	return 0;

}









