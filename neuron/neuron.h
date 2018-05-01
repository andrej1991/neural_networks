#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "../matrix/matrix.h"
#include "../opencl_setup.h"

#define SIGMOID 0
#define RELU 1
#define LEAKY_RELU 2

class Neuron{
    static int instance_count;
    cl_program sigmoid_program;
    cl_program sigmoid_derivate_program;
    cl_kernel sigmoid_kernel;
    cl_kernel sigmoid_derivate_kernel;
    cl_command_queue command_queue;
    int neuron_type;
    OpenclSetup *env;
    inline MatrixData sigmoid(MatrixData &input, int num_events=0, cl_event *wait_for_events=NULL);
    inline MatrixData sigmoid_derivate(MatrixData &input, int num_events=0, cl_event *wait_for_events=NULL);
    /*inline MatrixData relu(MatrixData &input);
    inline MatrixData leaky_relu(MatrixData &input);
    inline MatrixData relu_derivate(MatrixData &input);
    inline MatrixData leaky_relu_derivate(MatrixData &input);*/
    void load_neuron_operations_programs(cl_context *context, cl_device_id *deviceIds);
    public:
    Neuron(OpenclSetup *env, int neuron_type = SIGMOID);
    ~Neuron();
    MatrixData neuron(MatrixData &inputs, int num_events=0, cl_event *wait_for_events=NULL);
    MatrixData neuron_derivate(MatrixData &inputs, int num_events=0, cl_event *wait_for_events=NULL);
    void test();
};

#endif // NEURON_H_INCLUDED
