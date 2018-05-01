#define SIGMOID(x) (1.0f/(1.0f+exp(-1.0f*x)))

__kernel void sigmoid(__global float *input, __global float *output) 
{
    unsigned int index = get_global_id(0);
    output[index] = SIGMOID(input[index]);
}

/*__kernel void sigmoid_derivative(__global float *input, __global float *output) 
{
    unsigned int index = get_global_id(0);
    output[index] = SIGMOID(input[index])*(1-SIGMOID(input[index]));
}*/
