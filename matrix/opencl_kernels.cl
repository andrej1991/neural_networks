__kernel void transpose(__global float *input, __global float *output) 
{
    unsigned int group_size = get_local_size(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    output[group_size*local_id+group_id] = input[group_size*group_id+local_id];
}


__kernel void scalar_matrice_add(__global const float *A, __global const float *B, __global float *C) 
{
    int i = get_global_id(0);
    C[i] = A[i] + B[0];
}

__kernel void multiply(const int Arow, const int Bcol, const int Brow,
                      const __global float* A, const __global float* B, __global float* C)
{

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    float acc = 0.0f;
    for (int k=0; k<Brow; k++) {
        acc += A[globalRow*Brow + k] * B[k*Bcol + globalCol];
    }
 
    C[globalRow*Bcol + globalCol] = 1;
}

__kernel void multiply_with_transpose(const int Arow, const int Bcol, const int Brow,
                      const __global float* A, const __global float* B, __global float* C)
{
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    float acc = 0.0f;
    for (int k=0; k<Brow; k++) {
        acc += A[globalRow*Brow + k] * B[globalRow*Brow + k];
    }
 
    C[globalRow*Bcol + globalCol] = acc;
}

__kernel void transpose_ant_multiply(const int Arow, const int Bcol, const int Brow,
                      const __global float* A, const __global float* B, __global float* C)
{

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    float acc = 0.0f;
    for (int k=0; k<Brow; k++) {
        acc += A[k*Bcol + globalCol] * B[k*Bcol + globalCol];
    }
 
    C[globalRow*Bcol + globalCol] = acc;
}

__kernel void add(__global const float *A, __global const float *B, __global float *C) 
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}

__kernel void substract(__global const float *A, __global const float *B, __global float *C) 
{
    int i = get_global_id(0);
    C[i] = A[i] - B[i];
}

__kernel void hadamart_product(__global const float *A, __global const float *B, __global float *C) 
{
    int i = get_global_id(0);
    C[i] = A[i] * B[i];
}

__kernel void convolution(const int KernRow, const int KernCol, const int InpCol, const int OutpCol,
                      const __global float* inp,
                      const __global float* kern,
                      __global float* outp) {
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    float acc = 0.0f;
    int krow = KernRow-1;
    int kcol = KernCol-1;
    for(int i=0; i<KernRow; i++) 
    {
        for(int j=0; j<KernCol; j++)
        {
            acc += kern[(krow-i)*KernCol+kcol-j] * inp[(globalRow+i)*InpCol + globalCol+j];
        }
    }
 
    outp[globalRow*OutpCol + globalCol] = acc;
}

__kernel void fullconv(const int KernRow, const int KernCol, const int InpCol, const int InpRow, const int OutpCol,
                      const __global float* inp,
                      const __global float* kern,
                      __global float* outp) {
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    float acc = 0.0f;
    int krow = KernRow-1;
    int kcol = KernCol-1;
    for(int i=krow; i>=0; i--) 
    {
        for(int j=kcol; j>=0; j--)
        {
            if((globalRow >= i) && (globalCol >= j) && ((globalRow -i) < InpRow)&&((globalCol-j) < InpCol))
                acc += kern[i*KernCol+j] * inp[(globalRow-i)*InpCol + globalCol-j];
        }
    }
 
    outp[globalRow*OutpCol + globalCol] = acc;
}

__kernel void sameconv(const int KernRow, const int KernCol, const int DataCol,
                      const __global float* inp,
                      const __global float* kern,
                      __global float* outp) {
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int outpRow = get_global_size(0);
    const int outpCol = get_global_size(1);
 
    float acc = 0.0f;
    int krow = KernRow-1;
    int kcol = KernCol-1;
    int leftpadd = (KernCol-1)/2 + (KernCol-1)%2;
    int toppadd = (KernRow-1)/2 + (KernRow-1)%2;
    for(int i=0; i<KernRow; i++) 
    {
        for(int j=0; j<KernCol; j++)
        {
            if(((globalRow+i-toppadd)>=0)&&((globalCol+j-leftpadd)>=0)&&((globalRow+i-toppadd)<outpRow)&&((globalCol+j-leftpadd)<outpCol))
            acc += kern[(krow-i)*KernCol+kcol-j] * inp[(globalRow+i-toppadd)*DataCol + globalCol+j-leftpadd];
        }
    }
 
    outp[globalRow*DataCol + globalCol] = acc;
}

__kernel void zero(__global float* outp) {
    
    const int globalId = get_global_id(0);
 
    outp[globalId] = 0;
}

/*for(int i = kernel.row-1; i < input.row; i += stride)
        {
            for(int j = kernel.col-1; j < input.col; j += stride)
                {
                    helper = 0;
                    for(int k = kernel.row-1; k >= 0; k--)
                        {
                            for(int l = kernel.col-1; l >= 0; l--)
                                {
                                    helper += kernel[k][l] * input[i - k][j - l];
                                }
                        }
                    output[r][c] = helper;
                    c++;
                }
            r++;
            c = 0;
        }
*/

