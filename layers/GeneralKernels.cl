__kernel void update_weights(const float learning_rate, const float regularization_rate, __global const float *delta_weights, __global float *weights) 
{
    int i = get_global_id(0);
    float result = regularization_rate * weights[i] - learning_rate * delta_weights[i];
    weights[i] = result;
}
