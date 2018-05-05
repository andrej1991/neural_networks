#include "layers.h"

using namespace std;

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type, OpenclSetup *env):
    neuron(env, neuron_type), env(env)
{
    this->output = new MatrixData*[1];
    this->output[0] = new MatrixData(row, 1);
    this->output[0][0].copy_to_opencl_buffer(&(env->context));
    this->neuron_count = this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
    this->fmap = new Feature_map* [1];
    if(env == NULL)
    {
        cerr << "Invalid value for OpenclSetup in FullyConnected layers constructor!\n";
        throw exception();
    }
    this->fmap[0] = new Feature_map(row, prev_row, 1, row, env);
    this->update_weights_program = load_program("layers/layerspecific_kernels.cl", &(this->env->context), this->env->deviceIds);
    this->update_biases_program = load_program("layers/layerspecific_kernels.cl", &(this->env->context), this->env->deviceIds);
    cl_int errorcode;
    this->update_weights_kernel = clCreateKernel(this->update_weights_program, "update_weights", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_weights kernel\n";
        throw exception();
    }
    this->update_biases_kernel = clCreateKernel(this->update_biases_program, "update_biases", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_biases kernel\n";
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
    /*MatrixData multiplied(this->fmap[0][0].biases[0][0].get_row(), this->fmap[0][0].biases[0][0].get_col());
    multiplied.copy_to_opencl_buffer(&(this->env->context));*/
    MatrixData inputparam(this->fmap[0][0].biases[0][0].get_row(), this->fmap[0][0].biases[0][0].get_col());
    inputparam.copy_to_opencl_buffer(&(this->env->context));
    ///inputparam += (this->fmap[0][0].weights[0][0] * input[0][0] + this->fmap[0][0].biases[0][0]);
    cl_event event[2];
    this->fmap[0][0].mtxop[0].multiply(this->fmap[0][0].weights[0][0], input[0][0], inputparam, 0, NULL, &event[0]);
    clFinish(this->fmap[0][0].mtxop[0].command_queue);
    //this->fmap[0][0].mtxop[0].add_matrices(multiplied, this->fmap[0][0].biases[0][0], inputparam, 1, event, &event[1]);
    this->fmap[0][0].mtxop[0].add_matrices(inputparam, this->fmap[0][0].biases[0][0], inputparam, 0, NULL, &event[1]);

    this->neuron.activation(inputparam, this->output[0][0], 1, &event[1]);
}

void FullyConnected::sync_memory()
{
    int row = this->output[0][0].get_row();
    int col = this->output[0][0].get_col();
    size_t s = row * col * sizeof(float);
    clEnqueueReadBuffer(this->fmap[0][0].mtxop[0].command_queue, this->output[0][0].cl_mem_obj, CL_TRUE, 0, s, this->output[0][0].data, 0, NULL, NULL);
}

inline MatrixData FullyConnected::get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type)
{
    ///TODO create extra cernels for costfunction!!!
    MatrixData mtx(this->outputlen, 1);
    mtx.copy_to_opencl_buffer(&(this->env->context));
    MatrixData delta(this->outputlen, 1);
    delta.copy_to_opencl_buffer(&(this->env->context));
    MatrixData req_op = required_output;
    req_op.copy_to_opencl_buffer(&(this->env->context));
    MatrixData **output_derivate;
    switch(costfunction_type)
        {
        case QUADRATIC_CF:
            /*for(int i = 0; i < this->outputlen; i++)
                {
                    mtx[i][0] = (this->output[0][0])[i][0] - required_output[i][0];
                }*/
            this->fmap[0][0].mtxop[0].substract_matrices(this->output[0][0], req_op, mtx, 0, NULL, NULL);
            output_derivate = this->derivate_layers_output(input);
            clFinish(this->fmap[0][0].mtxop[0].command_queue);
            ///delta = hadamart_product(mtx, **output_derivate);
            this->fmap[0][0].mtxop[0].hadamart(mtx, output_derivate[0][0], delta);
            delete output_derivate[0];
            delete[] output_derivate;
            clEnqueueReadBuffer(this->fmap[0][0].mtxop[0].command_queue, delta.cl_mem_obj, CL_TRUE, 0, this->outputlen, delta.data, 0, NULL, NULL);
            return delta;
        case CROSS_ENTROPY_CF:
            switch(this->neuron_type)
                {
                case SIGMOID:
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            mtx[i][0] = (this->output[0][0])[i][0] - required_output[i][0];
                        }
                    return mtx;
                default:
                    output_derivate = this->derivate_layers_output(input);
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            delta[i][0] = ((output_derivate[0][0])[i][0] * ((this->output[0][0])[i][0] - required_output[i][0])) /
                                                    ((this->output[0][0])[i][0] * (1 - (this->output[0][0])[i][0]));
                        }
                    delete output_derivate[0];
                    delete[] output_derivate;
                    return delta;
                }
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline MatrixData** FullyConnected::derivate_layers_output(MatrixData **input)
{
    MatrixData **mtx;
    mtx = new MatrixData* [1];
    mtx[0] = new MatrixData(this->outputlen, 1);
    MatrixData inputparam(this->fmap[0][0].biases[0][0].get_row(), this->fmap[0][0].biases[0][0].get_col());
    cl_event event[2];
    this->fmap[0][0].mtxop[0].multiply(this->fmap[0][0].weights[0][0], input[0][0], inputparam, 0, NULL, &event[0]);
    clFinish(this->fmap[0][0].mtxop[0].command_queue);
    this->fmap[0][0].mtxop[0].add_matrices(inputparam, this->fmap[0][0].biases[0][0], inputparam);
    //clWaitForEvents(1, &event[1]);
    clFinish(this->fmap[0][0].mtxop[0].command_queue);
    ///inputparam = (this->fmap[0][0].weights[0][0] * input[0][0] + this->fmap[0][0].biases[0][0]);
    this->neuron.activation_derivate(inputparam, mtx[0][0]);
    return mtx;
}

/*cl_int errorcode;
size_t global_item_size = a.row*a.col;
size_t local_item_size = a.row;
errorcode = clSetKernelArg(this->matrice_add_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
errorcode = clSetKernelArg(this->matrice_add_kernel, 1, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
errorcode = clSetKernelArg(this->matrice_add_kernel, 2, sizeof(cl_mem), (void *)&(c.cl_mem_obj));
errorcode = clEnqueueNDRangeKernel(this->command_queue, this->matrice_add_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);*/

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    if((layer[0].get_fmap_count() != 1) + (layer[0].fmap[0][0].get_mapdepth() != 1))
        {
            cerr << "the fully connected layer must have only one set of weights!!!\n";
            throw exception();
        }
    int prev_outputlen = this->fmap[0][0].get_col();
    /*for(int j = 0; j < this->outputlen; j++)
        {
            (this->fmap[0][0].biases[0][0])[j][0] -= learning_rate * (layer[0].fmap[0][0].biases[0][0])[j][0];
            for(int k = 0; k < prev_outputlen; k++)
                {
                    (this->fmap[0][0].weights[0][0])[j][k] = regularization_rate * (this->fmap[0][0].weights[0][0])[j][k] - learning_rate * (layer[0].fmap[0][0].weights[0][0])[j][k];
                }
        }*/
    size_t global_item_size = prev_outputlen * this->outputlen;
    size_t local_item_size = this->outputlen;
    cl_int errorcode;
    errorcode = clSetKernelArg(this->update_weights_kernel, 0, sizeof(cl_mem), (void *)&(learning_rate));
    errorcode = clSetKernelArg(this->update_weights_kernel, 1, sizeof(cl_mem), (void *)&(regularization_rate));
    errorcode = clSetKernelArg(this->update_weights_kernel, 2, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->update_weights_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    global_item_size = this->outputlen;
    errorcode = clSetKernelArg(this->update_biases_kernel, 0, sizeof(cl_mem), (void *)&(learning_rate));
    errorcode = clSetKernelArg(this->update_biases_kernel, 1, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->update_biases_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    clFinish(this->fmap[0][0].mtxop[0].command_queue);
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
    MatrixData multiplied(next_layers_fmaps[0][0].weights[0][0].get_col(), delta[0][0].get_col()), **output_derivate;
    cl_event event[4];
    output_derivate = this->derivate_layers_output(input);
    ///multiplied = (next_layers_fmaps[0][0].weights[0][0].transpose()) * delta[0][0];
    //this->fmap[0][0].mtxop[0].multiply_with_transpose(next_layers_fmaps[0][0].weights[0][0], delta[0][0], multiplied, 0, NULL, &event[0]);
    this->fmap[0][0].mtxop[0].transpose_and_multiply(next_layers_fmaps[0][0].weights[0][0], delta[0][0], multiplied);
    clFinish(this->fmap[0][0].mtxop[0].command_queue);
    ///delta[0][0] = hadamart_product(multiplied, **output_derivate);
    //this->fmap[0][0].mtxop[0].hadamart(multiplied, output_derivate[0][0], delta[0][0], 1, &event[0], &event[1]);
    delete[] delta;
    delta = new MatrixData*[1];
    delta[0] = new MatrixData(this->outputlen, 1);
    delta[0][0].copy_to_opencl_buffer(&(this->env->context));
    this->fmap[0][0].mtxop[0].hadamart(multiplied, output_derivate[0][0], delta[0][0]);
    //this->fmap[0][0].mtxop[0].hadamart(multiplied, output_derivate[0][0], nabla[0][0].biases[0][0], 1, &event[0], &event[1]);
    this->fmap[0][0].mtxop[0].hadamart(multiplied, output_derivate[0][0], nabla[0][0].biases[0][0]);
    clFinish(this->fmap[0][0].mtxop[0].command_queue);
    ///nabla[0][0].weights[0][0] = delta[0][0] * input[0][0].transpose();
    this->fmap[0][0].mtxop[0].multiply_with_transpose(delta[0][0], input[0][0], nabla[0][0].weights[0][0]);
    clFinish(this->fmap[0][0].mtxop[0].command_queue);
    delete output_derivate[0];
    delete[] output_derivate;
    return delta;
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
