#include "layers.h"

using namespace std;

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type, OpenclSetup *env):
    neuron(env, neuron_type), env(env), neuron_count(row), outputlen(row)
{
    this->fmap = new Feature_map* [1];
    if(env == NULL)
    {
        cerr << "Invalid value for OpenclSetup in FullyConnected layers constructor!\n";
        throw exception();
    }
    this->fmap[0] = new Feature_map(row, prev_row, 1, row, env);
    this->output = new MatrixData*[1];
    this->output[0] = new MatrixData(this->outputlen, 1);
    this->output[0][0].copy_to_opencl_buffer(&(env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    this->output_derivative = new MatrixData*[1];
    this->output_derivative[0] = new MatrixData(this->outputlen, 1);
    this->output_derivative[0][0].copy_to_opencl_buffer(&(env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    this->output_error = new MatrixData*[1];
    this->output_error[0] = new MatrixData(this->outputlen, 1);
    this->output_error[0][0].copy_to_opencl_buffer(&(env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    for(int i=0; i<FullyConnectedFuncVarCount; i++)
    {
        this->function_variables[i] = NULL;
    }
    this->layer_type = FULLY_CONNECTED;
    this->fully_connected_program = load_program("layers/FullyConnectedKernels.cl", &(this->env->context), this->env->deviceIds);
    cl_int errorcode;
    this->update_weights_kernel = clCreateKernel(this->fully_connected_program, "update_weights", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_weights kernel\n";
        throw exception();
    }
    this->update_biases_kernel = clCreateKernel(this->fully_connected_program, "update_biases", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_biases kernel\n";
        throw exception();
    }
    this->get_act_input_kernel = clCreateKernel(this->fully_connected_program, "get_activation_input", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_biases kernel\n";
        throw exception();
    }
    this->get_layers_delta_kernel = clCreateKernel(this->fully_connected_program, "get_layers_delta", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_biases kernel\n";
        throw exception();
    }
    this->reg_rate = clCreateBuffer((this->env->context), CL_MEM_READ_WRITE, sizeof(float), NULL, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL buffer\n";
        throw exception();
    }
    this->learn_rate = clCreateBuffer((this->env->context), CL_MEM_READ_WRITE, sizeof(float), NULL, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL buffer\n";
        throw exception();
    }
}

FullyConnected::~FullyConnected()
{
    delete this->output[0];
    delete[] this->output;
    delete this->fmap[0];
    delete[] this->fmap;
}

inline void FullyConnected::layers_output(MatrixData **input)
{
    if(this->function_variables[0] == NULL)
    {
        this->function_variables[0] = new MatrixData(this->fmap[0][0].biases[0][0].get_row(), this->fmap[0][0].biases[0][0].get_col());
        this->function_variables[0]->copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    }
    ///inputparam += (this->fmap[0][0].weights[0][0] * input[0][0] + this->fmap[0][0].biases[0][0]);
    cl_int errorcode;
    cl_event event;
    size_t global = this->fmap[0][0].biases[0][0].row;
    errorcode = clSetKernelArg(this->get_act_input_kernel, 0, sizeof(int), (void*)&input[0][0].row);
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 1, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 2, sizeof(cl_mem), (void *)&(input[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 3, sizeof(cl_mem), (void *)&(this->fmap[0][0].biases[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 4, sizeof(cl_mem), (void *)&(this->function_variables[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->get_act_input_kernel, 1, NULL, &global, NULL, 0, NULL, &event);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened durring durring the multiplication of matrices\n" << errorcode << endl;
        throw exception();
    }
    this->neuron.activation(this->function_variables[0][0], this->output[0][0], 1, &event);
}

void FullyConnected::sync_memory()
{
    int row = this->output[0][0].get_row();
    int col = this->output[0][0].get_col();
    size_t s = row * col * sizeof(float);
    clEnqueueReadBuffer(this->fmap[0][0].mtxop[0].command_queue, this->output[0][0].cl_mem_obj, CL_TRUE, 0, s, this->output[0][0].data, 0, NULL, NULL);
}

inline MatrixData** FullyConnected::get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type)
{
    ///TODO create extra cernels for costfunction!!!
    /*MatrixData mtx(this->outputlen, 1);
    mtx.copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));*/
    if(this->function_variables[1] == NULL)
    {
        this->function_variables[1] = new MatrixData(this->outputlen, 1);
        this->function_variables[2] = new MatrixData;
        this->function_variables[1][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    }
    /*MatrixData delta(this->outputlen, 1);
    delta.copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));*/
    //MatrixData req_op = required_output;
    this->function_variables[2][0] = required_output;
    this->function_variables[2][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    MatrixData **output_derivate;
    cl_event events[2];
    switch(costfunction_type)
        {
        case QUADRATIC_CF:
            this->fmap[0][0].mtxop[0].substract_matrices(this->output[0][0], this->function_variables[2][0], this->function_variables[1][0], 0, NULL, &events[0]);
            this->derivate_layers_output(input);
            ///delta = hadamart_product(mtx, **output_derivate);
            this->fmap[0][0].mtxop[0].hadamart(this->function_variables[1][0], this->output_derivative[0][0], this->output_error[0][0], 1, &events[0], &events[1]);
            clWaitForEvents(1, &events[1]);
            return (this->output_error);
        /*case CROSS_ENTROPY_CF:
            switch(this->neuron_type)
                {
                case SIGMOID:
                    this->fmap[0][0].mtxop[0].substract_matrices(this->output[0][0], this->function_variables[2][0], this->output_error[0][0]);
                    clFinish(this->fmap[0][0].mtxop[0].command_queue);
                    return this->output_error;
                default:
                    this->derivate_layers_output(input);
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            (this->output_error[0][0])[i][0] = ((output_derivate[0][0])[i][0] * ((this->output[0][0])[i][0] - required_output[i][0])) /
                                                    ((this->output[0][0])[i][0] * (1 - (this->output[0][0])[i][0]));
                        }
                    clFinish(this->fmap[0][0].mtxop[0].command_queue);
                    return this->output_error;
                }*/
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline MatrixData** FullyConnected::derivate_layers_output(MatrixData **input)
{
    //MatrixData inputparam(this->fmap[0][0].biases[0][0].get_row(), this->fmap[0][0].biases[0][0].get_col());
    if(this->function_variables[0] == NULL)
    {
        this->function_variables[0] = new MatrixData(this->fmap[0][0].biases[0][0].get_row(), this->fmap[0][0].biases[0][0].get_col());
        this->function_variables[0]->copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    }
    cl_int errorcode;
    cl_event event;
    size_t global = this->fmap[0][0].biases[0][0].row;
    errorcode = clSetKernelArg(this->get_act_input_kernel, 0, sizeof(int), (void*)&input[0][0].row);
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 1, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 2, sizeof(cl_mem), (void *)&(input[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 3, sizeof(cl_mem), (void *)&(this->fmap[0][0].biases[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 4, sizeof(cl_mem), (void *)&(this->function_variables[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->get_act_input_kernel, 1, NULL, &global, NULL, 0, NULL, &event);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened durring durring the multiplication of matrices\n" << errorcode << endl;
        throw exception();
    }
    this->neuron.activation_derivate(this->function_variables[0][0], this->output_derivative[0][0], 1, &event);
    return (this->output_derivative);
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    if((layer[0].get_fmap_count() != 1) + (layer[0].fmap[0][0].get_mapdepth() != 1))
        {
            cerr << "the fully connected layer must have only one set of weights!!!\n";
            throw exception();
        }
    int prev_outputlen = this->fmap[0][0].get_col();
    size_t global_item_size = prev_outputlen * this->outputlen;
    size_t local_item_size = this->outputlen;
    cl_int errorcode;
    cl_event events[2];
    errorcode = clEnqueueWriteBuffer(this->fmap[0][0].mtxop[0].command_queue, this->reg_rate, CL_BLOCKING, 0, sizeof(float), (void*)&regularization_rate, 0, NULL, NULL);
    errorcode |= clEnqueueWriteBuffer(this->fmap[0][0].mtxop[0].command_queue, this->learn_rate, CL_BLOCKING, 0, sizeof(float), (void*)&learning_rate, 0, NULL, NULL);
    errorcode |= clSetKernelArg(this->update_weights_kernel, 0, sizeof(cl_mem), (void *)&(this->learn_rate));
    errorcode |= clSetKernelArg(this->update_weights_kernel, 1, sizeof(cl_mem), (void *)&(this->reg_rate));
    errorcode |= clSetKernelArg(this->update_weights_kernel, 2, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->update_weights_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &events[0]);

    global_item_size = this->outputlen;
    errorcode |= clSetKernelArg(this->update_biases_kernel, 0, sizeof(cl_mem), (void *)&(this->learn_rate));
    errorcode |= clSetKernelArg(this->update_biases_kernel, 1, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->update_biases_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &events[1]);
    clWaitForEvents(2, events);
}

inline void FullyConnected::remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void FullyConnected::add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void FullyConnected::set_input(MatrixData **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


inline MatrixData** FullyConnected::backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla,
                                          MatrixData **delta, int next_layers_fmapcount)
{
    ///TODO think through this function from mathematical perspective!!!
    if(next_layers_fmapcount != 1)
        {
            cerr << "Currently the fully connected layer can be followed only by fully connected layers!\n";
            throw exception();
        }
    if(this->function_variables[3] == NULL)
    {
        this->function_variables[3] = new MatrixData(this->outputlen, 1);
        this->function_variables[3]->copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    }
    cl_event events[2];
    ///MatrixData multiplied = (next_layers_fmaps[0][0].weights[0][0].transpose()) * delta[0][0];
    ///new_delta = hadamart(multiplied, this->output_derivative[0][0]);
    cl_int errorcode;
    size_t global_item_size = this->outputlen;
    errorcode = clSetKernelArg(this->get_layers_delta_kernel, 0, sizeof(int), (void *)&next_layers_fmaps[0][0].weights[0][0].col);
    errorcode |= clSetKernelArg(this->get_layers_delta_kernel, 1, sizeof(int), (void *)&(delta[0][0].row));
    errorcode |= clSetKernelArg(this->get_layers_delta_kernel, 2, sizeof(cl_mem), (void *)&(next_layers_fmaps[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_layers_delta_kernel, 3, sizeof(cl_mem), (void *)&(delta[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_layers_delta_kernel, 4, sizeof(cl_mem), (void *)&(this->output_derivative[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_layers_delta_kernel, 5, sizeof(cl_mem), (void *)&(this->function_variables[3][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_layers_delta_kernel, 6, sizeof(cl_mem), (void *)&(nabla[0][0].biases[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->get_layers_delta_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, &events[0]);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened durring calculating the delta of the layer\n" << errorcode << endl;
        throw exception();
    }
    this->fmap[0][0].mtxop[0].multiply_with_transpose(this->function_variables[3][0], input[0][0], nabla[0][0].weights[0][0], 1, &events[0], &events[1]);
    cl_int err = clWaitForEvents(1, &events[1]);
    return &(this->function_variables[3]);
}

inline MatrixData** FullyConnected::get_output()
{
    return this->output;
}

inline Feature_map** FullyConnected::get_feature_maps()
{
    return this->fmap;
}

inline short FullyConnected::get_layer_type()
{
    return this->layer_type;
}

inline int FullyConnected::get_output_row()
{
    return this->outputlen;
}

inline int FullyConnected::get_output_len()
{
    return this->outputlen;
}

inline int FullyConnected::get_output_col()
{
    return 1;
}

void FullyConnected::set_weights(MatrixData *w)
{
    this->fmap[0][0].weights[0][0] = *w;
}

void FullyConnected::set_biases(MatrixData *b)
{
    this->fmap[0][0].biases[0][0] = *b;
}

int FullyConnected::get_mapcount()
{
    return 1;
}

int FullyConnected::get_mapdepth()
{
    return 1;
}

int FullyConnected::get_weights_row()
{
    return this->fmap[0][0].get_row();
}

int FullyConnected::get_weights_col()
{
    return this->fmap[0][0].get_col();
}

void FullyConnected::store(std::ofstream &params)
{
    this->fmap[0][0].store(params);
}

void FullyConnected::load(std::ifstream &params)
{
    this->fmap[0][0].load(params);
}
