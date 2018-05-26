__kernel void softmax(const int output_len, const __global float* output_helper, __global float* output)
{
    int globalRow = get_global_id(0);
    float nominator = 0;
    for(int i = 0; i < output_len; i++)
    {
        nominator += output_helper[i];
    }
    output[globalRow] = output_helper[globalRow] / nominator;
}

__kernel void get_softmax_helper(const __global float* weighted_input, __global float* output_helper)
{
    int globalRow = get_global_id(0);
    output_helper[globalRow] = exp(weighted_input[globalRow]);
}

__kernel void softmax_derivative(const int output_len, const __global float* softmax_output, __global float* derivated_output)
{
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    if(globalRow == globalCol)
    {
        derivated_output[globalRow*output_len + globalCol] = softmax_output[globalRow] * (1.0f - softmax_output[globalCol]);
    }
    else
    {
        derivated_output[globalRow*output_len + globalCol] = -1.0f * softmax_output[globalRow] * softmax_output[globalCol];
    }
/*if(row == col)
{
    (mtx[0][0])[row][col] = (this->output[0][0])[row][0] * (1 - (this->output[0][0])[col][0]);
}
else
{
    (mtx[0][0])[row][col] = -1 * (this->output[0][0])[row][0] * (this->output[0][0])[col][0];
}*/
}
