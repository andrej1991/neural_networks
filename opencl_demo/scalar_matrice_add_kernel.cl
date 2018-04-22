__kernel void scalar_matrice_add(__global const float *A, __global const float *B, __global float *C) 
{
    int i = get_global_id(0);
    C[i] = A[i] + B[0];
}
