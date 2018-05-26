__kernel void ConvolutionAndAdd(const int KernRow, const int KernCol, const int InpCol, const int OutpCol,
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
 
    outp[globalRow*OutpCol + globalCol] += acc;
}

__kernel void FullConvAndAdd(const int KernRow, const int KernCol, const int InpCol, const int InpRow, const int OutpCol,
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
 
    outp[globalRow*OutpCol + globalCol] += acc;
}

__kernel void SameConvAndAdd(const int KernRow, const int KernCol, const int DataCol,
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
 
    outp[globalRow*DataCol + globalCol] += acc;
}
