__kernel void update_weights(const float learning_rate, const float regularization_rate, __global const float *delta_weights, __global float *weights) 
{
    int i = get_global_id(0);
    float result = regularization_rate * weights[i] - learning_rate * delta_weights[i];
    weights[i] = result;
}

__kernel void update_biases(const float learning_rate, __global const float *delta_biases, __global float *biases) 
{
    int i = get_global_id(0);
    float result = biases[i] - learning_rate * delta_biases[i];
    biases[i] = result;
}

__kernel void get_activation_input(const int InpRow, const __global float* weights,
                                   const __global float* input, const __global float* biases, __global float *output)
{
/*It assumes that the input is allways a column vector*/
    const int globalRow = get_global_id(0);
 
    float acc = 0.0f;
    for (int k=0; k<InpRow; k++) {
        acc += weights[globalRow*InpRow + k] * input[k];
    }
 
    output[globalRow] = acc + biases[globalRow];
}

__kernel void get_layers_delta(const int WeightsCol, const int DeltaRow, const __global float* weights,
                               const __global float* delta, const __global float* output_derivative,
                               __global float* layers_delta, __global float * delta_biases)
{
/*It assumes that the delta is allways a column vector*/
    const int globalRow = get_global_id(0);
 
    float acc = 0.0f;
    for (int k=0; k<DeltaRow; k++) {
        acc += weights[k*WeightsCol + globalRow] * delta[k];
    }
 
    layers_delta[globalRow] = acc * output_derivative[globalRow];
    delta_biases[globalRow] = acc * output_derivative[globalRow];
}
