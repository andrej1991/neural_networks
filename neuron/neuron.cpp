#include "neuron.h"
#include "../opencl_setup.h"
#include <iostream>

using namespace std;

int Neuron::instance_count = 0;
/*cl_program Neuron::sigmoid_program;
cl_program Neuron::sigmoid_derivate_program;*/

Neuron::Neuron(OpenclSetup *env, int neuron_type) : neuron_type(neuron_type), env(env)
{
    cout << "instantiating Neuron\n";
    //if(Neuron::instance_count == 0)
        this->load_neuron_operations_programs(&(env->context), env->deviceIds);
    instance_count++;
    cl_int errorcode;
    this->command_queue = clCreateCommandQueue(env->context, env->deviceIds[0], 0, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        std::cerr << "unable to create OpenCL command queue\n";
        throw exception();
    }
    this->sigmoid_kernel = clCreateKernel(this->sigmoid_program, "sigmoid", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        std::cerr << "unable to create OpenCL Neuron::sigmoid_program kernel\n";
        throw exception();
    }
    /*this->sigmoid_derivate_kernel = clCreateKernel(this->sigmoid_derivate_program, "sigmoid_derivative", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        std::cerr << "unable to create OpenCL Neuron::sigmoid_derivate_program kernel\n";
        throw exception();
    }*/
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
        clReleaseProgram(this->sigmoid_program);
        clReleaseProgram(this->sigmoid_derivate_program);

    }
}

void Neuron::load_neuron_operations_programs(cl_context *context, cl_device_id *deviceIds)
{
    cout << "loading neuron programs\n";
    this->sigmoid_program = load_program("neuron_kernels.cl",context, deviceIds);;
    this->sigmoid_derivate_program = load_program("neuron_kernels.cl",context, deviceIds);;
}

inline MatrixData Neuron::sigmoid(MatrixData &inputs, int num_events, cl_event *wait_for_events)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData ret(row, col);
    cl_int errorcode;
    size_t global_item_size = row*col;
    size_t local_item_size = col;
    cl_event event;
    errorcode = clSetKernelArg(this->sigmoid_kernel, 0, sizeof(cl_mem), (void *)&(inputs.cl_mem_obj));
    errorcode = clSetKernelArg(this->sigmoid_kernel, 0, sizeof(cl_mem), (void *)&(ret.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->sigmoid_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, &event);
    errorcode = clEnqueueReadBuffer(this->command_queue, ret.cl_mem_obj, CL_TRUE, 0, row*col * sizeof(float), ret.data, 1, &event, NULL);
    return ret;
}

inline MatrixData Neuron::sigmoid_derivate(MatrixData &inputs, int num_events, cl_event *wait_for_events)
{
    MatrixData s = sigmoid(inputs);
    int row = inputs.get_row();
    int col = inputs.get_col();
    MatrixData ret(row, col);
    cl_int errorcode;
    size_t global_item_size = row*col;
    size_t local_item_size = col;
    cl_event event;
    errorcode = clSetKernelArg(this->sigmoid_derivate_kernel, 0, sizeof(cl_mem), (void *)&(inputs.cl_mem_obj));
    errorcode = clSetKernelArg(this->sigmoid_derivate_kernel, 0, sizeof(cl_mem), (void *)&(ret.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->sigmoid_derivate_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, &event);
    errorcode = clEnqueueReadBuffer(this->command_queue, ret.cl_mem_obj, CL_TRUE, 0, row*col * sizeof(float), ret.data, 1, &event, NULL);
    return ret;
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

MatrixData Neuron::neuron(MatrixData &inputs, int num_events, cl_event *wait_for_events)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid(inputs, num_events, wait_for_events);
    /*case RELU:
        return this->relu(inputs);
    case LEAKY_RELU:
        return this->leaky_relu(inputs);*/
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}

MatrixData Neuron::neuron_derivate(MatrixData &inputs, int num_events, cl_event *wait_for_events)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid_derivate(inputs, num_events, wait_for_events);
    /*case RELU:
        return this->relu_derivate(inputs);
    case LEAKY_RELU:
        return this->leaky_relu_derivate(inputs);*/
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}
