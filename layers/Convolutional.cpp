#include "layers.h"

using namespace std;

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
    cl_int errorcode;
    this->fmap = new Feature_map* [map_count];
    this->outputs = new MatrixData* [map_count];
    this->convolution_helper = new MatrixData* [map_count];
    this->output_derivative = new MatrixData* [map_count];
    this->layers_delta = new MatrixData* [map_count];
    this->flattened_output = new MatrixData* [1];
    this->flattened_output[0] = new MatrixData(this->map_count * this->output_row * this->output_col, 1);
    for(int i = 0; i < map_count; i++)
        {
            fmap[i] = new Feature_map(this->kernel_row, this->kernel_col, input_channel_count);
            this->outputs[i] = new MatrixData(this->output_row, this->output_col);
            this->outputs[i][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[i][0].mtxop[0].command_queue));
            this->convolution_helper[i] = new MatrixData(this->output_row, this->output_col);
            this->convolution_helper[i][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[i][0].mtxop[0].command_queue));
            this->output_derivative[i] = new MatrixData(this->output_row, this->output_col);
            this->output_derivative[i][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[i][0].mtxop[0].command_queue));
            this->layers_delta[i] = new MatrixData(this->output_row, this->output_col);
            this->layers_delta[i][0].copy_to_opencl_buffer(&(this->env->context), &(this->fmap[i][0].mtxop[0].command_queue));
        }
}

Convolutional::~Convolutional()
{
    delete flattened_output[0];
    delete[] flattened_output;
    for(int i = 0; i < this->map_count; i++)
        {
            delete fmap[i];
            delete outputs[i];
        }
    delete[] fmap;
    delete[] outputs;
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

void Convolutional::get_2D_weights(int neuron_id, int fmap_id, MatrixData &kernel, Feature_map **next_layers_fmap)
{
    int kernelsize = kernel.get_row() * kernel.get_col();
    int starting_pos = kernelsize * fmap_id;
    int endpos = starting_pos + kernelsize;
    int index = starting_pos;
    for(int col = 0; col < kernel.get_col(); col++)
        {
            for(int row = 0; row < kernel.get_row(); row++)
                {
                    kernel[row][col] = (next_layers_fmap[0][0].weights[0][0])[neuron_id][index];
                    index++;
                }
        }
}

inline void calculate_delta_helper(MatrixData *padded_delta, MatrixData *delta_helper, MatrixData &kernel, MatrixData &helper)
{
    /*convolution(padded_delta[0],kernel, helper);
    delta_helper[0] += helper;*/
}

inline void delete_padded_delta(MatrixData **padded_delta, int limit)
{
    for(int i = 0; i < limit; i++)
            {
                delete padded_delta[i];
            }
        delete[] padded_delta;
}

inline MatrixData** Convolutional::backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **delta, int next_layers_fmapcount)
{
    /*MatrixData **layers_delta = new MatrixData* [this->map_count];
    MatrixData **output_derivate;
    output_derivate = this->derivate_layers_output(input);
    MatrixData **delta_helper;
    MatrixData **padded_delta;
    MatrixData helper(this->output_row, this->output_col);
    if(this->next_layers_type != CONVOLUTIONAL)
        {
            int next_layers_neuroncount = delta[0][0].get_row();
            padded_delta = new MatrixData* [next_layers_neuroncount];
            for(int i = 0; i < next_layers_neuroncount; i++)
                {
                    padded_delta[i] = new MatrixData;
                    (padded_delta[i][0])[0][0] = (delta[0][0])[i][0];
                    padded_delta[i][0] = padded_delta[i][0].zero_padd((this->output_row-1)/2,
                                                             (this->output_col-1)/2,
                                                             (this->output_row-1)/2,
                                                             (this->output_col-1)/2);
                }
            delta_helper = new MatrixData* [this->map_count];
            MatrixData kernel(this->output_row, this->output_col);
            for(int i = 0; i < this->map_count; i++)
                {
                    delta_helper[i] = new MatrixData(this->output_row, this->output_col);
                    for(int j = 0; j < next_layers_neuroncount; j++)
                        {
                            this->get_2D_weights(j, i, kernel, next_layers_fmaps);
                            calculate_delta_helper(padded_delta[j], delta_helper[i], kernel, helper);
                        }
                }
            delete_padded_delta(padded_delta, next_layers_neuroncount);
        }
    else
        {
            padded_delta = new MatrixData* [next_layers_fmapcount];
            for(int i = 0; i < next_layers_fmapcount; i++)
                {
                    padded_delta[i] = new MatrixData;
                    padded_delta[i][0] = delta[i][0];
                    padded_delta[i][0] = delta[i][0].zero_padd((next_layers_fmaps[i][0].weights[0][0].get_row()-1)/2,
                                                             (next_layers_fmaps[i][0].weights[0][0].get_col()-1)/2,
                                                             (next_layers_fmaps[i][0].weights[0][0].get_row()-1)/2,
                                                             (next_layers_fmaps[i][0].weights[0][0].get_col()-1)/2);
                }
            delta_helper = new MatrixData* [this->map_count];
            for(int i = 0; i < this->map_count; i++)
                {
                    delta_helper[i] = new MatrixData(this->output_row, this->output_col);
                    for(int j = 0; j < next_layers_fmapcount; j++)
                        {
                            calculate_delta_helper(padded_delta[j], delta_helper[i], next_layers_fmaps[j][0].weights[i][0], helper);
                        }
                }
            delete_padded_delta(padded_delta, next_layers_fmapcount);
        }
    for(int i = 0; i < this->map_count; i++)
        {
            layers_delta[i] = new MatrixData;
            layers_delta[i][0] = hadamart_product(delta_helper[i][0], output_derivate[i][0]);
            for(int j = 0; j < this->fmap[i][0].get_mapdepth(); j++)
                {
                     convolution(input[j][0], layers_delta[i][0], nabla[i][0].weights[j][0]);
                }
            delete output_derivate[i];
            delete delta_helper[i];
        }
    delete[] output_derivate;
    for(int i = 0; i < next_layers_fmapcount; i++)
        {
            //delete delta_helper[i];
            delete delta[i];
        }
    delete[] delta_helper;
    delete[] delta;
    return layers_delta;*/
}

void Convolutional::update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer)
{
    for(int i = 0; i < this->map_count; i++)
        {
            for(int j = 0; j < this->fmap[i][0].get_mapdepth(); j++)
                {
                    for(int row = 0; row < this->kernel_row; row++)
                        {
                            for(int col = 0; col < this->kernel_col; col++)
                                {
                                    (this->fmap[i][0].weights[j][0])[row][col] =
                                                    regularization_rate * (this->fmap[i][0].weights[j][0])[row][col] -
                                                    learning_rate * (layer[0].fmap[i][0].weights[j][0])[row][col];
                                }
                        }
                }
        }
}

inline void Convolutional::fulldepth_conv(MatrixData &helper, MatrixData &convolved, MatrixData **input, int map_index)
{
    /*for(int channel_index = 0; channel_index < this->fmap[map_index][0].get_mapdepth(); channel_index++)
        {
            convolution(input[channel_index][0], this->fmap[map_index][0].weights[channel_index][0], convolved, this->stride);
            helper += convolved;
        }
    helper+=(this->fmap[map_index][0].biases[0][0])[0][0];*/
}

inline void Convolutional::layers_output(MatrixData **input)
{
    /*MatrixData convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            this->fulldepth_conv(helper, convolved, input, map_index);
            this->outputs[map_index][0] = this->neuron.neuron(helper);
            helper.zero();
        }*/
}

inline MatrixData** Convolutional::get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

inline MatrixData** Convolutional::derivate_layers_output(MatrixData **input)
{
    /*MatrixData convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    MatrixData **ret = new MatrixData* [this->map_count];
    for(int i = 0; i < this->map_count; i++)
        {
            ret[i] = new MatrixData(this->output_row, this->output_col);
        }
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            this->fulldepth_conv(helper, convolved, input, map_index);
            ret[map_index][0] = this->neuron.neuron_derivate(helper);
            helper.zero();
        }
    return ret;*/
}

void Convolutional::flatten()
{
    ///TODO rewrite this if the feature maps can have different kernel size;
    int i = 0;
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            for(int col = 0; col < this->output_col; col++)
                {
                    for(int row = 0; row < this->output_row; row++)
                        {
                            (this->flattened_output[0][0])[i][0] = (this->outputs[map_index][0])[row][col];
                            i++;
                        }
                }
        }
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
    if(next_layers_type == FULLY_CONNECTED)
        {
            this->flatten();
            return this->flattened_output;
        }
    else
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
