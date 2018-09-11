/*! \file
    \brief Header file for GPU kernels.
*
*/

__global__ void readVec(float * dvector_in);

__global__ void readVecInt(int * dvector_in);

__global__ void calculate_secants(float * dsecants_out, float * dpoints_in, int * dsize_constants_in);

__global__ void calculate_col_norms(float * dprojected_secants, float * dsecant_norms, int * dsize_constants_in);

__global__ void switch_columns(float * matrix, int * dcolumn_switch_constants);

__global__ void shift_first_column(float * matrix, float * projection,float * shortest_secant, float * dalgo_constant);

__global__ void normalize_first_column(float * dproj, int * dsize_constants_in);

__global__ void make_identity(float * didentity_matrix, int * dsize_constants_in);
