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

	// Initialize key parameters of SAP algorithm
	int iterations;
	float step_size;

	// Query user for parameters
	cout << "Number of iterations (a positive integer): \n";
	cin >> iterations;
	cout << "Step size (real number between 0 and .5): \n";
	cin >> step_size;

	// Initialize parameters to characterize data
	int input_dim;
	int numb_points;
	int proj_dim;

	/*--------------- Begin importing data --------------*/

	// Read data points from a text file
	ifstream inFile;
	inFile.open("example_data_points.txt");
	if (!inFile){
		cerr << "Unable to open file with data points.\n";
	exit(1);
	}

	// First two rows give the input data dimension and 
	// the number of points
	inFile >> input_dim;
	inFile >> numb_points;

	// Initialize array to store data points
	float * h_points_in = new float[input_dim*numb_points];
	
	// Read in entries for data point from text file
	float x;
	int i = 0;
	while (i < input_dim*numb_points){
		inFile >> x;
		h_points_in[i] = x;
		i = i+1;
	}	

	inFile.close();
	
	// Read in initial projection
	inFile.open("initial_projection.txt");
	if (!inFile){
		cerr << "Unable to open initial projection file.\n";
	exit(1);
	}

	// Projection dimension is stored as first entry to text file
	inFile >> proj_dim;

	// Initialize array to store initial projection
	float * h_proj = new float[input_dim*proj_dim];

	// Read in entries of initial projection
	i = 0;
	while (i < proj_dim*input_dim){
		inFile >> x;
		h_proj[i] = x;
		i = i+1;
	}	

	inFile.close();

	// Create file to store the smallest secant norms
	float h_smallest_secant_norms[iterations];

	// Call the SAP algorithm
	SAP(input_dim, numb_points, h_points_in, proj_dim, h_proj, h_smallest_secant_norms, iterations, step_size);

	// Open file to store output projection
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









