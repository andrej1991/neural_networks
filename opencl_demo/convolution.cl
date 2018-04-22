__kernel void convolution(const int KernRow, const int KernCol, const int InpCol, const int OutpCol,
                      const __global float* inp,
                      const __global float* kern,
                      __global float* outp) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    // Compute a single element (loop over K)
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
 
    // Store the result*/
    outp[globalRow*OutpCol + globalCol] = acc;
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
