__kernel void transpose(__global float *input, __global float *output) 
{
    unsigned int group_size = get_local_size(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    output[group_size*local_id+group_id] = input[group_size*group_id+local_id];
}

