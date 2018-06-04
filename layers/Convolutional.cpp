#include "layers.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

Convolutional::Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col,
                             int map_count, int neuron_type, int next_layers_type, Padding &p, OpenclSetup *env, int stride):
                    input_row(input_row), input_col(input_col), kernel_row(kern_row), kernel_col(kern_col), map_count(map_count), neuron_type(neuron_type),
                    next_layers_type(next_layers_type), pad(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), neuron(env, neuron_type),
                    stride(stride), env(env)
{
    if(stride != 1)
        {
            cerr << "counting with stride different than 1 is not implemented yet!";
            throw exception();
        }
    this->output_row = input_row - kern_row + 1;
    this->output_col = input_col - kern_col + 1;
    this->layer_type = CONVOLUTIONAL;
    this->convolutional_program = load_program("layers/ConvolutionalKernels.cl", &(this->env->context), this->env->deviceIds);
    this->general_program = load_program("layers/GeneralKernels.cl", &(this->env->context), this->env->deviceIds);
    cl_int errorcode;
    this->fmap = new Feature_map* [map_count];
    this->outputs = new MatrixData* [map_count];
    this->convolution_helper = new MatrixData* [map_count];
    this->output_derivative = new MatrixData* [map_count];
    this->layers_delta = new MatrixData* [map_count];
    this->layers_delta_helper = new MatrixData* [map_count];
    this->delta_helper = clCreateBuffer(this->env->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &errorcode);
    fmap[0] = new Feature_map(this->kernel_row, this->kernel_col, input_channel_count, this->map_count, -1, env);
    this->outputs[0] = new MatrixData(this->map_count * this->output_row * this->output_col, 1);
    this->outputs[0][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    this->convolution_helper[0] = new MatrixData(this->map_count * this->output_row * this->output_col, 1);
    this->convolution_helper[0][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    this->output_derivative[0] = new MatrixData(this->map_count * this->output_row * this->output_col, 1);
    this->output_derivative[0][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    this->layers_delta[0] = new MatrixData(this->map_count * this->output_row * this->output_col, 1);
    this->layers_delta[0][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    this->layers_delta_helper[0] = new MatrixData(this->map_count * this->output_row * this->output_col, 1);
    this->layers_delta_helper[0][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[0][0].mtxop[0].command_queue));
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL buffer. The error is:" << errorcode << endl;
        throw exception();
    }
    this->fulldepth_conv_kernel = clCreateKernel(this->convolutional_program, /*"FullDepthConvolution"*/"GetDeltaHelper", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL ConvolutionAndAdd kernel\n";
        throw exception();
    }
    this->fullconv_and_add_kernel = clCreateKernel(this->convolutional_program, "FullConvAndAdd", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL FullConvAndAdd kernel\n";
        throw exception();
    }
    this->update_weights_kernel = clCreateKernel(this->general_program, "update_weights", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL update weights kernel\n";
        throw exception();
    }
    this->fulldepth_fullconv_kernel = clCreateKernel(this->convolutional_program, /*"GetDeltaHelper"*/"FullDepthConvolution", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL FullDepthFullConv kernel\n";
        throw exception();
    }
    this->delta_weight_kernel = clCreateKernel(this->convolutional_program, "GettingDeltaWeights", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL GettingDeltaWeights kernel\n";
        throw exception();
    }
}

Convolutional::~Convolutional()
{
    /*delete flattened_output[0];
    delete[] flattened_output;
    for(int i = 0; i < this->map_count; i++)
        {
            delete fmap[i];
            delete outputs[i];
        }
    delete[] fmap;
    delete[] outputs;*/
}

void Convolutional::sync_memory()
{
    int row, col;
    cl_event events[this->map_count];
    for(int i=0; i<this->map_count; i++)
    {
        row = this->outputs[i][0].get_row();
        col = this->outputs[i][0].get_col();
        size_t s = row * col * sizeof(float);
        clEnqueueReadBuffer(this->fmap[i][0].mtxop[0].command_queue, this->outputs[i][0].cl_mem_obj, CL_FALSE, 0, s, this->outputs[i][0].data, 0, NULL, &events[i]);
    }
    clWaitForEvents(this->map_count, events);
}


inline void delete_padded_delta(MatrixData **padded_delta, int limit)
{
    for(int i = 0; i < limit; i++)
            {
                delete padded_delta[i];
            }
        delete[] padded_delta;
}

void print_execution_time(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2)
{
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "The execution time was: " << duration << "us" << endl;;
}

inline MatrixData** Convolutional::backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **delta, int next_layers_fmapcount)
{
    this->derivate_layers_output(input);
    MatrixData **delta_helper;
    MatrixData helper(this->output_row, this->output_col);
    cl_int errorcode;
    cl_event write_events[this->map_count];
    //const size_t global[3] = {this->layers_delta_helper[0][0].row, this->layers_delta_helper[0][0].col, this->map_count};
    const size_t global[3] = {this->output_row, this->output_col, this->map_count};
    //const size_t global[3] = {1, 1, 1};
    cl_event event1, event2, event3;
    if(this->next_layers_type != CONVOLUTIONAL)
    {
        int delta_rc = 1;
        int next_layers_neuroncount = delta[0][0].get_row();
        int delta_row = delta[0][0].row/next_layers_fmapcount;
        int nextl_kern_row = next_layers_fmaps[0][0].get_row();
        int nextl_kern_col = next_layers_fmaps[0][0].get_col();
        errorcode = clSetKernelArg(this->fulldepth_fullconv_kernel, 0, sizeof(int), (void*)&(nextl_kern_row));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 1, sizeof(int), (void*)&(nextl_kern_col));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 2, sizeof(int), (void*)&(delta_rc));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 3, sizeof(int), (void*)&(delta_rc));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 4, sizeof(int), (void*)&(next_layers_neuroncount));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 5, sizeof(cl_mem), (void *)&(delta[0][0].cl_mem_obj));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 6, sizeof(cl_mem), (void *)&(next_layers_fmaps[0][0].weights[0][0].cl_mem_obj));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 7, sizeof(cl_mem), (void *)&(this->layers_delta_helper[0][0].cl_mem_obj));
        errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->fulldepth_fullconv_kernel, 3, NULL, global, NULL, 0, NULL, &event1);
        if(errorcode != CL_SUCCESS)
        {
            cerr << "Some error happened durring full depth convolution\n the errorcode is: " << errorcode << endl;
            throw exception();
        }
    }
    else
    {
        int nextl_kern_row = next_layers_fmaps[0][0].get_row();
        int nextl_kern_col = next_layers_fmaps[0][0].get_col();
        int delta_row = this->output_row;//delta[0][0].row/next_layers_fmapcount;
        int delta_col = this->output_col;//delta[0][0].col;
        errorcode = clSetKernelArg(this->fulldepth_fullconv_kernel, 0, sizeof(int), (void*)&(nextl_kern_row));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 1, sizeof(int), (void*)&(nextl_kern_col));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 2, sizeof(int), (void*)&(delta_col));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 3, sizeof(int), (void*)&(delta_row));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 4, sizeof(int), (void*)&(next_layers_fmapcount));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 5, sizeof(cl_mem), (void *)&(delta[0][0].cl_mem_obj));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 6, sizeof(cl_mem), (void *)&(next_layers_fmaps[0][0].weights[0][0].cl_mem_obj));
        errorcode |= clSetKernelArg(this->fulldepth_fullconv_kernel, 7, sizeof(cl_mem), (void *)&(this->layers_delta_helper[0][0].cl_mem_obj));
        errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->fulldepth_fullconv_kernel, 3, NULL, global, NULL, 0, NULL, &event1);
        if(errorcode != CL_SUCCESS)
        {
            cerr << "Some error happened durring full depth convolution\n the errorcode is: " << errorcode << endl;
            throw exception();
        }
    }
    this->fmap[0][0].mtxop[0].hadamart(this->layers_delta_helper[0][0], this->output_derivative[0][0], layers_delta[0][0], 1, &event1, &event2);
    clWaitForEvents(1, &event2);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    errorcode = clSetKernelArg(this->delta_weight_kernel, 0, sizeof(int), (void*)&(this->output_row));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 1, sizeof(int), (void*)&(this->output_col));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 2, sizeof(int), (void*)&(this->input_col));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 3, sizeof(int), (void*)&(this->input_row));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 4, sizeof(int), (void*)&(this->map_count));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 5, sizeof(cl_mem), (void *)&(input[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 6, sizeof(cl_mem), (void *)&(layers_delta[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->delta_weight_kernel, 7, sizeof(cl_mem), (void *)&(nabla[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->delta_weight_kernel, 3, NULL, global, NULL, 1, &event2, &event3);
    if(errorcode != CL_SUCCESS)
        {
            cerr << "Some error happened durring calculating the delta weights of Convolutional layer\n the errorcode is: " << errorcode << endl;
            throw exception();
        }
    clWaitForEvents(1, &event3);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
        cout << "delta helper";
        print_execution_time(t1, t2);
    return this->layers_delta;
}

void Convolutional::update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer)
{
    cl_int errorcode;
    cl_event event;
    size_t global_item_size = this->fmap[0][0].weights[0][0].row * this->fmap[0][0].weights[0][0].col;
    size_t local_item_size = this->fmap[0][0].weights[0][0].row;
    errorcode = clSetKernelArg(this->update_weights_kernel, 0, sizeof(float), (void *)&(learning_rate));
    errorcode |= clSetKernelArg(this->update_weights_kernel, 1, sizeof(float), (void *)&(regularization_rate));
    errorcode |= clSetKernelArg(this->update_weights_kernel, 2, sizeof(cl_mem), (void *)&(layer[0].fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(this->update_weights_kernel, 3, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, this->update_weights_kernel, 1, NULL, &global_item_size, NULL/*&local_item_size*/, 0, NULL, &event);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened while updating the weights of the Convolutional layer\n" << errorcode << endl;
        throw exception();
    }
    clWaitForEvents(1, &event);
}

inline void Convolutional::fulldepth_conv(MatrixData **input, cl_kernel *opencl_kernel)
{
    cl_event eventset1[this->map_count];
    cl_int errorcode;
    const size_t global[3] = {this->output_row, this->output_col, this->map_count};
    cl_event eventset2[this->map_count];
    cl_event event;
    int mapdepth = this->fmap[0][0].get_mapdepth();
    errorcode = clSetKernelArg(*opencl_kernel, 0, sizeof(int), (void*)&(this->kernel_row));
    errorcode |= clSetKernelArg(*opencl_kernel, 1, sizeof(int), (void*)&(this->kernel_col));
    errorcode |= clSetKernelArg(*opencl_kernel, 2, sizeof(int), (void*)&(this->input_col));
    errorcode |= clSetKernelArg(*opencl_kernel, 3, sizeof(int), (void*)&(this->input_row));
    errorcode |= clSetKernelArg(*opencl_kernel, 4, sizeof(int), (void*)&(mapdepth));
    errorcode |= clSetKernelArg(*opencl_kernel, 5, sizeof(cl_mem), (void *)&(input[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(*opencl_kernel, 6, sizeof(cl_mem), (void *)&(this->fmap[0][0].weights[0][0].cl_mem_obj));
    errorcode |= clSetKernelArg(*opencl_kernel, 7, sizeof(cl_mem), (void *)&(this->convolution_helper[0][0].cl_mem_obj));
    errorcode |= clEnqueueNDRangeKernel(this->fmap[0][0].mtxop[0].command_queue, *opencl_kernel, 3, NULL, global, NULL, 0, NULL, &event);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "Some error happened durring full depth convolution\n the errorcode is: " << errorcode << endl;
        throw exception();
    }
    clWaitForEvents(1, &event);
}

inline void Convolutional::layers_output(MatrixData **input)
{
    this->fulldepth_conv(input, &(this->fulldepth_conv_kernel));
    this->neuron.activation(this->convolution_helper[0][0], this->outputs[0][0]);

}

inline MatrixData** Convolutional::get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

inline MatrixData** Convolutional::derivate_layers_output(MatrixData **input)
{
    this->fulldepth_conv(input, &(this->fulldepth_conv_kernel));
    this->neuron.activation_derivate(this->convolution_helper[0][0], this->output_derivative[0][0]);
}

void Convolutional::flatten()
{
    ///TODO rewrite this if the feature maps can have different kernel size;
    int dst_offset = 0;
    int size_to_copy;
    cl_event events[this->map_count];
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        size_to_copy = sizeof(float)*(this->outputs[map_index][0].row)*(this->outputs[map_index][0].col);
        clEnqueueCopyBuffer(this->fmap[map_index][0].mtxop[0].command_queue, this->outputs[map_index][0].cl_mem_obj, this->outputs[0][0].cl_mem_obj,
                            0, dst_offset, size_to_copy, 0, NULL, &events[map_index]);
        dst_offset += size_to_copy;
    }
    clWaitForEvents(this->map_count, events);
}


inline void Convolutional::remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

inline void Convolutional::add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

void Convolutional::set_input(MatrixData **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}

inline MatrixData** Convolutional::get_output()
{
    //print_mtx_list(outputs, this->map_count);
    //cout << "---------------------------" << endl;
        return this->outputs;

}

inline Feature_map** Convolutional::get_feature_maps()
{
    return this->fmap;
}

inline short Convolutional::get_layer_type()
{
    return CONVOLUTIONAL;
}

inline int Convolutional::get_output_row()
{
    return this->output_row;
}

inline int Convolutional::get_output_len()
{
    return (this->output_row * this->output_col * this->map_count);
}

inline int Convolutional::get_output_col()
{
    return this->output_col;
}

void Convolutional::set_weights(MatrixData *w)
{
    ;
}

void Convolutional::set_biases(MatrixData *b)
{
    ;
}

int Convolutional::get_mapcount()
{
    return this->map_count;
}

int Convolutional::get_mapdepth()
{
    return this->fmap[0][0].get_mapdepth();
}

int Convolutional::get_weights_row()
{
    return this->kernel_row;
}

int Convolutional::get_weights_col()
{
    return this->kernel_col;
}

void Convolutional::store(std::ofstream &params)
{
    for(int i = 0; i < this->map_count; i++)
        {
            this->fmap[i][0].store(params);
        }
}
void Convolutional::load(std::ifstream &params)
{
    for(int i = 0; i < this->map_count; i++)
        {
            this->fmap[i][0].load(params);
        }
}
