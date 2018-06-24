#include "layers.h"
#include <math.h>

using namespace std;

Softmax::Softmax(int row, int col, OpenclSetup *env): FullyConnected(row, col, -1, env)
{
    this->layer_type = SOFTMAX;
    this->softmax_program = load_program("layers/SoftmaxKernels.cl", &(this->env->context), this->env->deviceIds);
    cl_int errorcode;
    this->softmax_kernel = clCreateKernel(this->softmax_program, "softmax", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_weights kernel\n";
        throw exception();
    }
    this->softmax_derivative_kernel = clCreateKernel(this->softmax_program, "softmax_derivative", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_weights kernel\n";
        throw exception();
    }
    this->softmax_helper_kernel = clCreateKernel(this->softmax_program, "get_softmax_helper", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update_weights kernel\n";
        throw exception();
    }
}

Softmax::~Softmax()
{
    clReleaseKernel(this->softmax_kernel);
    clReleaseKernel(this->softmax_derivative_kernel);
    clReleaseKernel(this->softmax_helper_kernel);
    clReleaseProgram(this->softmax_program);
}

inline MatrixData** Softmax::backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **next_layers_error, int next_layers_fmapcount)
{
    cerr << "Softamx layer can only be an output layer!!!\n";
    throw exception();
}

inline void Softmax::layers_output(MatrixData **input)
{
    if(this->function_variables[0] == NULL)
    {
        this->function_variables[0] = new MatrixData(this->outputlen, 1);
        this->function_variables[0]->copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
        this->function_variables[1] = new MatrixData(this->outputlen, 1);
        this->function_variables[1]->copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    }
    ///inputparam += (this->fmap[0][0].weights[0][0] * input[0][0] + this->fmap[0][0].biases[0][0]);
    cl_int errorcode;
    cl_event events[3];
    size_t global = this->outputlen;
    errorcode = clSetKernelArg(this->get_act_input_kernel, 0, sizeof(int), (void*)&input[0][0].row);
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 1, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 2, sizeof(cl_mem), (void *)&(input[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 3, sizeof(cl_mem), (void *)&(this->fmap[0][0].biases[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->get_act_input_kernel, 4, sizeof(cl_mem), (void *)&(this->function_variables[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->get_act_input_kernel, 1, NULL, &global, NULL, 0, NULL, &events[0]);

    errorcode |= clSetKernelArg(this->softmax_helper_kernel, 0, sizeof(cl_mem), (void*)&(this->function_variables[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->softmax_helper_kernel, 1, sizeof(cl_mem), (void *)&(this->function_variables[1][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->softmax_helper_kernel, 1, NULL, &global, NULL, 1, &events[0], &events[1]);

    errorcode |= clSetKernelArg(this->softmax_kernel, 0, sizeof(int), (void*)&(this->outputlen));
    errorcode |= clSetKernelArg(this->softmax_kernel, 1, sizeof(cl_mem), (void*)&(this->function_variables[1][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->softmax_kernel, 2, sizeof(cl_mem), (void *)&(this->output[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->softmax_kernel, 1, NULL, &global, NULL, 1, &events[1], &events[2]);

    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened while calculatingt the output of the Softmax layer\n" << errorcode << endl;
        throw exception();
    }
    clWaitForEvents(1, &events[2]);
}

inline MatrixData** Softmax::get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type)
{
    if(this->function_variables[2] == NULL)
    {
        this->function_variables[2] = new MatrixData(this->outputlen, 1);
        this->function_variables[2]->copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
        this->function_variables[3] = new MatrixData;
        //this->function_variables[3][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    }
    this->function_variables[3][0] = required_output;
    this->function_variables[3][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    //clEnqueueWriteBuffer(this->fmap[0][0].mtxop[0].command_queue, this->function_variables[3][0].cl_mem_obj, CL_TRUE, 0, required_output.row*required_output.col, (void*)required_output.data, 0, NULL, NULL);
    switch(costfunction_type)
        {
        /*case QUADRATIC_CF:
            for(int i = 0; i < this->outputlen; i++)
                {
                    mtx[i][0] = (this->output[0][0])[i][0] - required_output[i][0];
                }
            output_derivate = this->derivate_layers_output(input);
            delta = output_derivate[0][0] * mtx;
            delete output_derivate[0];
            delete[] output_derivate;
            return delta;*/
        case LOG_LIKELIHOOD_CF:
            cl_event event;
            this->fmap[0][0].mtxop[0].substract_matrices(this->output[0][0], this->function_variables[3][0], this->output_error[0][0], 0, NULL, &event);
            clWaitForEvents(1, &event);
            return this->output_error;
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline MatrixData** Softmax::derivate_layers_output(MatrixData **input)
{
    this->layers_output(input);
    cl_int errorcode;
    cl_event event;
    const size_t global[2] = { (size_t)this->outputlen, (size_t)this->outputlen };

    errorcode = clSetKernelArg(this->softmax_derivative_kernel, 0, sizeof(int), (void*)&(this->outputlen));
    errorcode |= clSetKernelArg(this->softmax_derivative_kernel, 1, sizeof(cl_mem), (void *)&(this->output[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->softmax_derivative_kernel, 2, sizeof(cl_mem), (void *)&(this->output_derivative[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->softmax_derivative_kernel, 2, NULL, global, NULL, 0, NULL, &event);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened durring valid convolution\n";
        throw exception();
    }
    clWaitForEvents(1, &event);
    return this->output_derivative;
}


///these are inherited from Softmax layer
/*void Softmax::update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer)
{
    ;
}

inline void Softmax::remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void Softmax::add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void Softmax::set_input(MatrixData **input)
{
    ;
}

inline MatrixData** Softmax::get_output()
{
    ;
}

inline Feature_map** Softmax::get_feature_maps()
{
    ;
}

inline short Softmax::get_layer_type()
{
    ;
}

inline int Softmax::get_output_len()
{
    ;
}

inline int Softmax::get_output_row()
{
    ;
}

inline int Softmax::get_output_col()
{
    ;
}

void Softmax::set_weights(MatrixData *w)
{
    ;
}

void Softmax::set_biases(MatrixData *b)
{
    ;
}

int Softmax::get_mapcount()
{
    ;
}

int Softmax::get_mapdepth()
{
    ;
}

int Softmax::get_weights_row()
{
    ;
}

int Softmax::get_weights_col()
{
    ;
}*/


