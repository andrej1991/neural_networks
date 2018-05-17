#include "neuron.h"
#include "../opencl_setup.h"
#include <iostream>

using namespace std;

int Neuron::instance_count = 0;
cl_program Neuron::neuron_program;

Neuron::Neuron(OpenclSetup *env, int neuron_type) : neuron_type(neuron_type), env(env)
{
    if(Neuron::instance_count == 0)
        this->load_neuron_operations_programs(&(env->context), env->deviceIds);
    instance_count++;
    cl_int errorcode;
    this->command_queue = clCreateCommandQueue(env->context, env->deviceIds[0], 0, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        std::cerr << "unable to create OpenCL command queue\n";
        throw exception();
    }
    this->sigmoid_kernel = clCreateKernel(Neuron::neuron_program, "sigmoid", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        std::cerr << "unable to create OpenCL Neuron::sigmoid_program kernel\n";
        throw exception();
    }
    this->sigmoid_derivate_kernel = clCreateKernel(Neuron::neuron_program, "sigmoid_derivative", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        std::cerr << "unable to create OpenCL Neuron::sigmoid_derivate_program kernel\n";
        throw exception();
    }
}

Neuron::~Neuron()
{
    clReleaseKernel(this->sigmoid_kernel);
    clReleaseKernel(this->sigmoid_derivate_kernel);
    clFlush(this->command_queue);
    clFinish(this->command_queue);
    clReleaseCommandQueue(this->command_queue);
    Neuron::instance_count--;
    if(Neuron::instance_count == 0)
    {
        clReleaseProgram(Neuron::neuron_program);

    }
}

void Neuron::load_neuron_operations_programs(cl_context *context, cl_device_id *deviceIds)
{
    Neuron::neuron_program = load_program("neuron/neuron_kernels.cl",context, deviceIds);;
}

inline void Neuron::sigmoid(MatrixData &inputs, MatrixData &outputs, int num_events, cl_event *wait_for_events)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    cl_int errorcode;
    size_t global_item_size = row*col;
    size_t local_item_size = col;
    cl_event event;
    errorcode = clSetKernelArg(this->sigmoid_kernel, 0, sizeof(cl_mem), (void *)&(inputs.cl_mem_obj));
    errorcode = clSetKernelArg(this->sigmoid_kernel, 1, sizeof(cl_mem), (void *)&(outputs.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->sigmoid_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, &event);
    errorcode = clWaitForEvents(1, &event);
}

inline void Neuron::sigmoid_derivate(MatrixData &inputs, MatrixData &outputs, int num_events, cl_event *wait_for_events)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData s(row, col);
    this->sigmoid(inputs, s);
    cl_int errorcode;
    size_t global_item_size = row*col;
    size_t local_item_size = col;
    cl_event event;
    errorcode = clSetKernelArg(this->sigmoid_derivate_kernel, 0, sizeof(cl_mem), (void *)&(inputs.cl_mem_obj));
    errorcode = clSetKernelArg(this->sigmoid_derivate_kernel, 1, sizeof(cl_mem), (void *)&(outputs.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->sigmoid_derivate_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, &event);
    errorcode = clWaitForEvents(1, &event);
}

/*inline MatrixData Neuron::relu(MatrixData &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs[i][j] > 0)
                        output[i][j] = inputs[i][j];
                    else
                        output[i][j] = 0;
                }
        }
    return output;
}
inline MatrixData Neuron::relu_derivate(MatrixData &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs[i][j] > 0)
                        output[i][j] = 1;
                    else
                        output[i][j] = 0;
                }
        }
    return output;
}

inline MatrixData Neuron::leaky_relu(MatrixData &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs[i][j] > 0)
                        output[i][j] = inputs[i][j];
                    else
                        output[i][j] = 0.001*inputs[i][j];
                }
        }
    return output;
}
inline MatrixData Neuron::leaky_relu_derivate(MatrixData &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs[i][j] > 0)
                        output[i][j] = 1;
                    else
                        output[i][j] = 0.001;
                }
        }
    return output;
}*/

void Neuron::activation(MatrixData &inputs, MatrixData &outputs, int num_events, cl_event *wait_for_events)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        this->sigmoid(inputs, outputs, num_events, wait_for_events);
        break;
    /*case RELU:
        this->relu(inputs);
        break;
    case LEAKY_RELU:
        this->leaky_relu(inputs);
        break;*/
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}

void Neuron::activation_derivate(MatrixData &inputs, MatrixData &outputs, int num_events, cl_event *wait_for_events)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        this->sigmoid_derivate(inputs, outputs, num_events, wait_for_events);
        break;
    /*case RELU:
        this->relu_derivate(inputs);
        break;
    case LEAKY_RELU:
        this->leaky_relu_derivate(inputs);
        break;*/
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}
