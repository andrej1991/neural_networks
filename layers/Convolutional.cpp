#include <math.h>
#include <string.h>
#include "Convolutional.h"

conv_output_helper::conv_output_helper(int threadcnt, int row, int col)
{
    this->threadcount = threadcnt;
    this->convolved = new Matrix* [threadcnt];
    this->helper = new Matrix* [threadcnt];
    for(int i = 0; i < threadcnt; i++)
    {
        this->convolved[i] = new Matrix(row, col);
        this->helper[i] = new Matrix(row, col);
    }
}

conv_output_helper::~conv_output_helper()
{
    for(int i = 0; i < this->threadcount; i++)
    {
        delete this->convolved[i];
        delete this->helper[i];
    }
    delete[] this->convolved;
    delete[] this->helper;
}

conv_backprop_helper::conv_backprop_helper(int threadcnt, int row, int col)
{
    this->padded_delta = new Matrix*** [threadcnt];
    this->dilated = new Matrix [threadcnt];
    this->helper = new Matrix* [threadcnt];
    this->kernel = new Matrix* [threadcnt];
    this->threadcount = threadcnt;
    this->layer_count = new int *[threadcnt];
    this->layer_count[0] = NULL;
    for(int i = 0; i < threadcnt; i++)
    {
        this->helper[i] = new Matrix(row, col);
        this->kernel[i] = new Matrix(row, col);
        this->padded_delta[i] = NULL;
        this->layer_count[i] = 0;
    }
}

conv_backprop_helper::~conv_backprop_helper()
{
    for(int i = 0; i < this->threadcount; i++)
    {
        delete this->helper[i];
        delete this->kernel[i];
        this->delete_padded_delta(i);
    }
    delete[] this->padded_delta;
    delete[] dilated;
    delete helper;
    delete kernel;
}

void conv_backprop_helper::delete_padded_delta(int threadindx)
{
    if(this->padded_delta[threadindx] == NULL)
    {
        return;
    }
    for(int i = 0; i < this->outputcount; i++)
    {
        for(int j = 0; j < this->layer_count[threadindx][i]; j++)
        {
            delete this->padded_delta[threadindx][i][j];
        }
        delete[] this->padded_delta[threadindx][i];
        this->padded_delta[threadindx][i] = NULL;
        this->layer_count[threadindx][i] = 0;
    }
    //delete[] this->padded_delta;
    //this->padded_delta = NULL;
}

void conv_backprop_helper::set_padded_delta_1d(Matrix **delta, int next_layers_neuroncount, int top, int right, int bottom, int left, int threadcnt)
{
    /*this->layer_count[threadcnt] = next_layers_neuroncount;
    padded_delta[threadcnt] = new Matrix* [next_layers_neuroncount];
    for(int j = 0; j < next_layers_neuroncount; j++)
    {
        padded_delta[threadcnt][j] = new Matrix;
        padded_delta[threadcnt][j][0].data[0][0] = delta[0][0].data[j][0];
        padded_delta[threadcnt][j][0] = padded_delta[threadcnt][j][0].zero_padd(top, right, bottom, left);
    }*/
}

void conv_backprop_helper::set_padded_delta_2d(Matrix ***delta, std::vector<int> sends_output, Layer **network_layers, int threadcnt)
{
    if(this->layer_count[0] == NULL)
    {
        for(int i = 0; i < this->threadcount; i++)
        {
            this->layer_count[i] = new int[sends_output.size()];
        }
        this->outputcount = sends_output.size();
    }
    for(int i : sends_output)
    {
        int next_layers_fmapcount = network_layers[i]->get_mapcount();
        this->layer_count[threadcnt][i] = next_layers_fmapcount;
        padded_delta[threadcnt][i] = new Matrix* [next_layers_fmapcount];
        Feature_map** next_layers_fmaps = network_layers[i]->get_feature_maps();
        for(int j = 0; j < next_layers_fmapcount; j++)
        {
            padded_delta[threadcnt][i][j] = new Matrix;
            dilated[threadcnt] = delta[i][j][0].dilate(static_cast<Convolutional*>(network_layers[i])->get_vertical_stride(), static_cast<Convolutional*>(network_layers[i])->get_horizontal_stride());
            padded_delta[threadcnt][i][j][0] = dilated[threadcnt].zero_padd((next_layers_fmaps[j]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[j]->weights[0]->get_col()-1)/2,
                                                     (next_layers_fmaps[j]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[j]->weights[0]->get_col()-1)/2);
        }
    }
}

void conv_backprop_helper::zero(int threadid)
{
    for(int j = 0; j < this->outputcount; j++)
    {
        for(int i = 0; i < this->layer_count[threadid][j]; i++)
            this->padded_delta[threadid][j][i]->zero();
    }
}

int conv_backprop_helper::get_layercount(int threadidx)
{
    return this->layer_count[threadidx][0];
}

Convolutional::Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, int my_index_, Padding &p, int vertical_stride, int horizontal_stride):
                    input_row(input_row), input_col(input_col), kernel_row(kern_row), kernel_col(kern_col), map_count(map_count), vertical_stride(vertical_stride), horizontal_stride(horizontal_stride), my_index(my_index_),
                    next_layers_type(next_layers_type), pad(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), neuron(neuron_type), neuron_type(neuron_type), input_channel_count(input_channel_count)
{
    this->output_row = (input_row - kern_row + vertical_stride) / vertical_stride;
    this->output_col = (input_col - kern_col + horizontal_stride) / horizontal_stride;
    if((this->output_row <= 0) || (this->output_col <= 0))
    {
        cerr << "You are using too big kernel or strides!" << endl;
        throw exception();
    }
    if(((input_row - kern_row + vertical_stride) % vertical_stride != 0) || ((input_col - kern_col + horizontal_stride) % horizontal_stride != 0))
    {
        cerr << "The stride or the size of the kernel is too big!" << endl;
        throw exception();
    }
    this->threadcount = 1;
    this->layer_type = CONVOLUTIONAL;
    this->fmap = new Feature_map* [map_count];
    double deviation = 2.0/sqrt(kern_row * kern_col);
    double mean = 0.0;
    if(neuron_type == SIGMOID)
    {
        mean = 0.5;
    }
    else if(neuron_type == RELU || neuron_type == LEAKY_RELU)
    {
        mean = deviation;
    }
    for(int i = 0; i < map_count; i++)
    {
        fmap[i] = new Feature_map(this->kernel_row, this->kernel_col, 1, this->output_row, this->output_col);
        fmap[i]->initialize_weights(deviation);
        fmap[i]->initialize_biases(deviation);
    }
    this->build_outputs_and_errors();
}

Convolutional::~Convolutional()
{
    for(int i = 0; i < this->map_count; i++)
    {
        delete fmap[i];
    }
    delete[] fmap;
    this->destory_outputs_and_erros();
}

void Convolutional::destory_outputs_and_erros()
{
    for(int i = 0; i < this->threadcount; i++)
    {
        for(int j = 0; j < this->map_count; j++)
        {
            delete this->outputs[i][j];
            delete this->output_derivative[i][j];
            delete this->layers_delta[i][j];
            delete this->layers_delta_helper[i][j];
        }
        delete[] this->outputs[i];
        delete[] this->flattened_output[i];
        delete[] this->output_derivative[i];
        delete[] this->layers_delta[i];
        delete[] this->layers_delta_helper[i];
    }
    delete[] this->outputs;
    delete[] this->flattened_output;
    delete[] this->output_derivative;
    delete[] this->layers_delta;
    delete[] this->layers_delta_helper;

    delete this->feedforward_helpter;
    delete this->backprop_helper;
}

void Convolutional::build_outputs_and_errors()
{
    this->feedforward_helpter = new conv_output_helper(this->threadcount, this->output_row, this->output_col);
    this->backprop_helper = new conv_backprop_helper(this->threadcount, this->output_row, this->output_col);

    this->outputs = new Matrix** [this->threadcount];
    this->flattened_output = new Matrix** [this->threadcount];
    this->output_derivative = new Matrix** [this->threadcount];
    this->layers_delta = new Matrix** [this->threadcount];
    this->layers_delta_helper = new Matrix** [this->threadcount];
    for(int i = 0; i < this->threadcount; i++)
    {
        this->outputs[i] = new Matrix* [this->map_count];
        this->flattened_output[i] = new Matrix* [1];
        this->output_derivative[i] = new Matrix* [this->map_count];
        this->layers_delta[i] = new Matrix* [this->map_count];
        this->layers_delta_helper[i] = new Matrix* [this->map_count];
        this->flattened_output[i][0] = new Matrix(this->map_count * this->output_row * this->output_col, 1);
        for(int j = 0; j < this->map_count; j++)
        {
            this->outputs[i][j] = new Matrix(this->output_row, this->output_col);
            this->output_derivative[i][j] = new Matrix(this->output_row, this->output_col);
            this->layers_delta[i][j] = new Matrix(this->output_row, this->output_col);
            this->layers_delta_helper[i][j] = new Matrix(this->output_row, this->output_col);
        }
    }
}

void Convolutional::get_2D_weights(int neuron_id, int fmap_id, Matrix &kernel, Feature_map **next_layers_fmap)
{
    int kernelsize = kernel.get_row() * kernel.get_col();
    int starting_pos = kernelsize * fmap_id;
    memcpy(kernel.dv, &(next_layers_fmap[0]->weights[0]->data[neuron_id][starting_pos]), kernelsize*sizeof(double));
}

/*void calculate_delta_helper(Matrix *padded_delta, Matrix *delta_helper, Matrix &kernel, Matrix &helper)
{
    cross_correlation(padded_delta[0], kernel, helper, 1, 1);
    delta_helper[0] += helper;
}*/

Matrix** Convolutional::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***delta, int threadindex)
{
    Feature_map** next_layers_fmaps;
    /*if(this->next_layers_type != POOLING)
    {
        next_layers_fmaps = next_layer->get_feature_maps();
    }*/
    int next_layers_fmapcount;// = next_layer->get_mapcount();
    this->derivate_layers_output(input, threadindex);
    /*if(this->next_layers_type == FULLY_CONNECTED or this->next_layers_type == SOFTMAX)
    {
        int next_layers_neuroncount = delta[0]->get_row();
        if(this->backprop_helper->get_layercount(threadindex) != next_layers_neuroncount)
        {
            this->backprop_helper->delete_padded_delta(threadindex);
        }
        if(this->backprop_helper->padded_delta[threadindex] == NULL)
        {
            this->backprop_helper->set_padded_delta_1d(delta, next_layers_neuroncount, (this->output_row-1)/2, (this->output_col-1)/2,
                                                       (this->output_row-1)/2, (this->output_col-1)/2, threadindex);
        }
        //this->backprop_helper->zero(threadindex);
        for(int i = 0; i < this->map_count; i++)
        {
            this->layers_delta_helper[threadindex][i][0].zero();
            for(int j = 0; j < next_layers_neuroncount; j++)
            {
                this->get_2D_weights(j, i, this->backprop_helper->kernel[threadindex][0], next_layers_fmaps);
                //calculate_delta_helper(this->backprop_helper->padded_delta[threadindex][j], layers_delta_helper[threadindex][i],
                //                       this->backprop_helper->kernel[threadindex][0], this->backprop_helper->helper[threadindex][0]);
                //cross_correlation(padded_delta[0], kernel, helper, 1, 1);
                full_depth_cross_correlation(this->backprop_helper->padded_delta[threadindex][j][0],
                                            this->backprop_helper->kernel[threadindex][0],
                                            this->layers_delta_helper[threadindex][i][0],
                                            1, 1);
            }
        }
        //this->backprop_helper->zero(threadindex);
    }
    else if (this->next_layers_type == CONVOLUTIONAL)
    {
    if(this->backprop_helper->get_layercount(threadindex) != next_layers_fmapcount)
    {
        this->backprop_helper->delete_padded_delta(threadindex);
    }
    if(this->backprop_helper->padded_delta[threadindex] == NULL)
    {
        this->backprop_helper->set_padded_delta_2d(delta, this->sends_output_to_, this->network_layers, threadindex);
    }*/
    this->backprop_helper->delete_padded_delta(threadindex);
    this->backprop_helper->set_padded_delta_2d(delta, this->sends_output_to_, this->network_layers, threadindex);
    //this->backprop_helper->zero(threadindex);
    for(int i = 0; i < this->map_count; i++)
    {
        this->layers_delta_helper[threadindex][i][0].zero();
        for(int next_layer_index : this->sends_output_to_)
        {
            next_layers_fmaps = this->network_layers[next_layer_index]->get_feature_maps();
            next_layers_fmapcount = this->network_layers[next_layer_index]->get_mapcount();
            for(int j = 0; j < next_layers_fmapcount; j++)
            {
                //calculate_delta_helper(this->backprop_helper->padded_delta[threadindex][j], this->layers_delta_helper[threadindex][i],
                  //                     next_layers_fmaps[j]->weights[i][0], this->backprop_helper->helper[threadindex][0]);
                full_depth_cross_correlation(this->backprop_helper->padded_delta[threadindex][next_layer_index][j][0],
                                            next_layers_fmaps[j]->weights[0][0],
                                            this->layers_delta_helper[threadindex][i][0],
                                            1, 1);
            }
        }
    }
        //this->backprop_helper->zero(threadindex);
    //}
    for(int i = 0; i < this->map_count; i++)
    {
        if(this->next_layers_type != POOLING)
        {
            this->layers_delta[threadindex][i][0] = hadamart_product(this->layers_delta_helper[threadindex][i][0], this->output_derivative[threadindex][i][0]);
        }
        else
        {
            this->layers_delta[threadindex][i][0].zero();
            for(int delta_index : this->sends_output_to_)
            {
                this->layers_delta[threadindex][i][0] += hadamart_product(delta[delta_index][i][0], this->output_derivative[threadindex][i][0]);
            }
        }
        this->backprop_helper->dilated[threadindex] = this->layers_delta[threadindex][i][0].dilate(this->vertical_stride, this->horizontal_stride);
        nabla[i]->weights[0][0].zero();
        for(int j = 0; j < this->input_channel_count; j++)
        {
            full_depth_cross_correlation(input[j][0], this->backprop_helper->dilated[threadindex], nabla[i]->weights[0][0], this->vertical_stride, this->horizontal_stride);
        }
        nabla[i]->biases[0][0] = this->layers_delta[threadindex][i][0];
    }
    return this->layers_delta[threadindex];
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *gradient)
{
    for(int i = 0; i < this->map_count; i++)
    {
        for(int row = 0; row < this->fmap[i]->biases[0][0].get_row(); row++)
        {
            for(int col = 0; col < this->fmap[i]->biases[0][0].get_col(); col++)
            {
                this->fmap[i]->biases[0][0].data[row][col] -= learning_rate * this->fmap[i]->biases[0][0].data[row][col];
            }
        }
        for(int j = 0; j < this->fmap[i]->get_mapdepth(); j++)
        {
            for(int row = 0; row < this->kernel_row; row++)
            {
                for(int col = 0; col < this->kernel_col; col++)
                {
                    this->fmap[i]->weights[j]->data[row][col] =
                                    regularization_rate * this->fmap[i]->weights[j]->data[row][col] -
                                    learning_rate * gradient->fmap[i]->weights[j]->data[row][col];
                }
            }
        }
    }
}

void Convolutional::fulldepth_conv(Matrix &helper, Matrix &convolved, Matrix **input, int map_index)
{
    for(int channel_index = 0; channel_index < this->input_channel_count; channel_index++)
    {
        //convolution(input[channel_index][0], this->fmap[map_index]->weights[channel_index][0], convolved, this->vertical_stride, this->horizontal_stride);
        //helper += convolved;
        full_depth_convolution(input[channel_index][0], this->fmap[map_index]->weights[0][0], helper, this->vertical_stride, this->horizontal_stride);
    }
    helper+=this->fmap[map_index]->biases[0][0];
}

void Convolutional::layers_output(Matrix **inpput, int threadindex)
{
    this->feedforward_helpter->helper[threadindex]->zero();
    for(int i : this->gets_input_from_)
    {
        for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            this->fulldepth_conv(this->feedforward_helpter->helper[threadindex][0], this->feedforward_helpter->convolved[threadindex][0], this->network_layers[i]->get_output(threadindex), map_index);
            this->neuron.neuron(this->feedforward_helpter->helper[threadindex][0], this->outputs[threadindex][map_index][0]);
            this->feedforward_helpter->helper[threadindex]->zero();
        }
    }
}

Matrix* Convolutional::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

Matrix** Convolutional::derivate_layers_output(Matrix **input, int threadindex)
{
    this->feedforward_helpter->helper[threadindex]->zero();
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {   ///allready calculated, might happen that in RNN case it's going to be needed again to be calculated
        //this->fulldepth_conv(this->feedforward_helpter->helper[threadindex][0], this->feedforward_helpter->convolved[threadindex][0], input, map_index);
        this->neuron.neuron_derivative(this->feedforward_helpter->helper[threadindex][0], output_derivative[threadindex][map_index][0]);
        this->feedforward_helpter->helper[threadindex]->zero();
    }
    return output_derivative[threadindex];
}

void Convolutional::flatten(int threadindex)
{
    int output_size = this->output_row * this->output_col;
    int output_size_in_bytes = output_size * sizeof(double);
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        memcpy(&(this->flattened_output[threadindex][0]->dv[map_index*output_size]), this->outputs[threadindex][map_index]->dv, output_size_in_bytes);
    }
}

void Convolutional::set_threadcount(int threadcnt, vector<Matrix***> inputs_)
{
    this->destory_outputs_and_erros();

    this->threadcount = threadcnt;

    this->build_outputs_and_errors();
}

int Convolutional::get_threadcount()
{
    return this->threadcount;
}

void Convolutional::set_input(Matrix **input, int threadindex)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}

Matrix** Convolutional::get_output(int threadindex)
{
    /*if(this->next_layers_type == FULLY_CONNECTED)
    {
        this->flatten(threadindex);
        return this->flattened_output[threadindex];
    }
    else*/
        return this->outputs[threadindex];
}

void Convolutional::create_connections(vector<int> input_from, vector<int> output_to)
{
    ///TODO some error checking
    this->gets_input_from_ = input_from;
    this->sends_output_to_ = output_to;
}

const vector<int>& Convolutional::gets_input_from() const
{
    return this->gets_input_from_;
}

const vector<int>& Convolutional::sends_output_to() const
{
    return this->sends_output_to_;
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

int Convolutional::get_mapcount()
{
    return this->map_count;
}

int Convolutional::get_mapdepth()
{
    return this->fmap[0]->get_mapdepth();
}

int Convolutional::get_weights_row()
{
    return this->kernel_row;
}

int Convolutional::get_weights_col()
{
    return this->kernel_col;
}

int Convolutional::get_vertical_stride()
{
    return this->vertical_stride;
}

int Convolutional::get_horizontal_stride()
{
    return this->horizontal_stride;
}

void Convolutional::store(std::ofstream &params)
{
    for(int i = 0; i < this->map_count; i++)
    {
        this->fmap[i]->store(params);
    }
}
void Convolutional::load(std::ifstream &params)
{
    for(int i = 0; i < this->map_count; i++)
    {
        this->fmap[i]->load(params);
    }
}

void Convolutional::set_graph_information(Layer **network_layers, int my_index)
{
    this->my_index = my_index;
    this->network_layers = network_layers;
}
