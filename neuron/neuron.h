#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "../matrix/matrix.h"
#include "../opencl_setup.h"

#define SIGMOID 0
#define RELU 1
#define LEAKY_RELU 2

class Neuron{
    static int instance_count;
    static cl_program neuron_program;
    cl_kernel sigmoid_kernel;
    cl_kernel sigmoid_derivate_kernel;
    cl_command_queue command_queue;
    int neuron_type;
    OpenclSetup *env;
    inline void sigmoid(MatrixData &input, MatrixData &output, int num_events=0, cl_event *wait_for_events=NULL);
    inline void sigmoid_derivate(MatrixData &input, MatrixData &output, int num_events=0, cl_event *wait_for_events=NULL);
    /*inline MatrixData relu(MatrixData &input, MatrixData &output);
    inline MatrixData leaky_relu(MatrixData &input, MatrixData &output);
    inline MatrixData relu_derivate(MatrixData &input, MatrixData &output);
    inline MatrixData leaky_relu_derivate(MatrixData &input, MatrixData &output);*/
    void load_neuron_operations_programs(cl_context *context, cl_device_id *deviceIds);
    public:
    Neuron(OpenclSetup *env, int neuron_type = SIGMOID);
    ~Neuron();
    void activation(MatrixData &inputs, MatrixData &outputs, int num_events=0, cl_event *wait_for_events=NULL);
    void activation_derivate(MatrixData &inputs, MatrixData &outputs, int num_events=0, cl_event *wait_for_events=NULL);
    void test();
};

#endif // NEURON_H_INCLUDED
