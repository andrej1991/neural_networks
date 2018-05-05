__kernel void update_weights(__global const float *learning_rate, __global const float *regularization_rate, __global float *weights) 
{
    int i = get_global_id(0);
    float result = regularization_rate[0] * weights[i] - learning_rate[0] * weights[i];
    weights[i] = result;
}

__kernel void update_biases(__global const float *learning_rate, __global float *biases) 
{
    int i = get_global_id(0);
    float result = biases[i] - learning_rate[0] * biases[i];
    biases[i] = result;
}
