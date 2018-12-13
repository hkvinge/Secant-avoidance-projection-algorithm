/*! \file
*  \brief Header file for custom GPU kernels used in the SAP algorithm
*
*/

__global__ void readVec(float * d_vector);

__global__ void readVecInt(int * d_vector);

__global__ void calculate_secants(float * d_secants, float * dpoints_in, int * d_int_constants);

__global__ void calculate_col_norms(float * d_matrix, float * d_col_norms, int * d_int_constants);

__global__ void switch_columns(float * d_matrix, int * d_column_switch_indices);

__global__ void shift_first_column(float * d_matrix, float * projectioni, float * d_new_column, float * d_algo_constants);

__global__ void normalize_first_column(float * d_matrix, int * d_int_constants);

__global__ void make_identity(float * d_matrix, int * d_int_constants);









