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
    this->padded_delta = new Matrix** [threadcnt];
    this->dilated = new Matrix [threadcnt];
    this->helper = new Matrix* [threadcnt];
    this->kernel = new Matrix* [threadcnt];
    this->threadcount = threadcnt;
    for(int i = 0; i < threadcnt; i++)
    {
        this->helper[i] = new Matrix(row, col);
        this->kernel[i] = new Matrix(row, col);
        this->padded_delta[i] = NULL;
    }
}

conv_backprop_helper::~conv_backprop_helper()
{
    this->delete_padded_delta();
    for(int i = 0; i < this->threadcount; i++)
    {
        delete this->helper[i];
        delete this->kernel[i];
    }
    delete[] dilated;
    delete helper;
    delete kernel;
}

void conv_backprop_helper::delete_padded_delta()
{
    if(this->padded_delta[0] == NULL)
    {
        return;
    }
    for(int i = 0; i < this->threadcount; i++)
    {
        for(int j = 0; j < this->layer_count; j++)
        {
            delete this->padded_delta[i][j];
        }
        delete[] this->padded_delta[i];
    }
    delete[] this->padded_delta;
    this->padded_delta = NULL;
}

void conv_backprop_helper::set_padded_delta_1d(Matrix **delta, int next_layers_neuroncount, int top, int right, int bottom, int left, int threadcnt)
{
    //int next_layers_neuroncount = delta[0]->get_row();
    padded_delta[threadcnt] = new Matrix* [next_layers_neuroncount];
    this->layer_count = next_layers_neuroncount;
    for(int j = 0; j < next_layers_neuroncount; j++)
    {
        padded_delta[threadcnt][j] = new Matrix;
        padded_delta[threadcnt][j][0].data[0][0] = delta[0][0].data[j][0];
        padded_delta[threadcnt][j][0] = padded_delta[threadcnt][j][0].zero_padd(top, right, bottom, left);
                                                 /*(this->output_col-1)/2,
                                                 (this->output_row-1)/2,
                                                 (this->output_col-1)/2);*/
    }
}

void conv_backprop_helper::set_padded_delta_2d(Matrix **delta, int next_layers_fmapcount, Layer *next_layer, int threadcnt)
{
    padded_delta[threadcnt] = new Matrix* [next_layers_fmapcount];
    Feature_map** next_layers_fmaps = next_layer->get_feature_maps();
    this->layer_count = next_layers_fmapcount;
    for(int j = 0; j < next_layers_fmapcount; j++)
    {
        padded_delta[threadcnt][j] = new Matrix;
        dilated[threadcnt] = delta[j][0].dilate(static_cast<Convolutional*>(next_layer)->get_vertical_stride(), static_cast<Convolutional*>(next_layer)->get_horizontal_stride());
        padded_delta[threadcnt][j][0] = dilated[threadcnt].zero_padd((next_layers_fmaps[j]->weights[0]->get_row()-1)/2,
                                                 (next_layers_fmaps[j]->weights[0]->get_col()-1)/2,
                                                 (next_layers_fmaps[j]->weights[0]->get_row()-1)/2,
                                                 (next_layers_fmaps[j]->weights[0]->get_col()-1)/2);
    }
}

void conv_backprop_helper::zero(int threadid)
{
    for(int i = 0; i < this->layer_count; i++)
    {
        this->padded_delta[threadid][i]->zero();
    }
}

Convolutional::Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int vertical_stride, int horizontal_stride):
                    input_row(input_row), input_col(input_col), kernel_row(kern_row), kernel_col(kern_col), map_count(map_count), vertical_stride(vertical_stride), horizontal_stride(horizontal_stride),
                    next_layers_type(next_layers_type), pad(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), neuron(neuron_type), neuron_type(neuron_type)
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
    //this->feedforward_helpter = new conv_output_helper(this->threadcount, this->output_row, this->output_col);
    //this->backprop_helper = new conv_backprop_helper(this->threadcount, this->output_row, this->output_col);
    this->layer_type = CONVOLUTIONAL;
    this->fmap = new Feature_map* [map_count];
    /*this->outputs = new Matrix** [1];
    this->outputs[0] = new Matrix* [map_count];
    this->output_derivative = new Matrix** [1];
    this->output_derivative[0] = new Matrix* [map_count];
    this->layers_delta = new Matrix** [1];
    this->layers_delta[0] = new Matrix* [map_count];
    this->layers_delta_helper = new Matrix** [1];
    this->layers_delta_helper[0] = new Matrix* [map_count];
    this->flattened_output = new Matrix** [1];
    this->flattened_output[0] = new Matrix* [1];
    this->flattened_output[0][0] = new Matrix(this->map_count * this->output_row * this->output_col, 1);*/
    double deviation = sqrt(2.0/(kern_row * kern_col));
    for(int i = 0; i < map_count; i++)
    {
        fmap[i] = new Feature_map(this->kernel_row, this->kernel_col, input_channel_count);
        fmap[i]->initialize_weights(deviation);
        fmap[i]->initialize_biases();
        /*this->outputs[0][i] = new Matrix(this->output_row, this->output_col);
        this->output_derivative[0][i] = new Matrix(this->output_row, this->output_col);
        this->layers_delta[0][i] = new Matrix(this->output_row, this->output_col);
        this->layers_delta_helper[0][i] = new Matrix(this->output_row, this->output_col);*/
    }
    this->build_outputs_and_errors();
}

Convolutional::~Convolutional()
{
    //delete flattened_output[0];
    //delete[] flattened_output;
    for(int i = 0; i < this->map_count; i++)
    {
        delete fmap[i];
        /*delete outputs[i];
        delete output_derivative[i];
        delete layers_delta[i];
        delete layers_delta_helper[i];*/
    }
    delete[] fmap;
    /*delete[] outputs;
    delete[] output_derivative;
    delete[] layers_delta;
    delete[] layers_delta_helper;*/
    this->destory_outputs_and_erros();
}

void Convolutional::destory_outputs_and_erros()
{
    for(int i = 0; i < this->threadcount; i++)
    {
        for(int j = 0; j < this->map_count; j++)
        {
            delete this->outputs[i][j];
            //delete this->flattened_output[i][j];
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
    int index = starting_pos;
    memcpy(kernel.dv, &(next_layers_fmap[0]->weights[0]->data[neuron_id][starting_pos]), kernelsize*sizeof(double));
}

void calculate_delta_helper(Matrix *padded_delta, Matrix *delta_helper, Matrix &kernel, Matrix &helper)
{
    //convolution(padded_delta[0],kernel, helper);
    cross_correlation(padded_delta[0], kernel, helper, 1, 1);
    delta_helper[0] += helper;
}

/*void delete_padded_delta(Matrix **padded_delta, int limit)
{
    for(int i = 0; i < limit; i++)
    {
        delete padded_delta[i];
    }
    delete[] padded_delta;
}*/

Matrix** Convolutional::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **delta, int threadindex)
{
    Feature_map** next_layers_fmaps;
    if(this->next_layers_type != POOLING)
    {
        next_layers_fmaps = next_layer->get_feature_maps();
    }
    int next_layers_fmapcount = next_layer->get_mapcount();
    this->derivate_layers_output(input, threadindex);
    //Matrix **padded_delta;
    //Matrix helper(this->output_row, this->output_col);
    //Matrix dilated;
    if(this->next_layers_type == FULLY_CONNECTED or this->next_layers_type == SOFTMAX)
    {
        int next_layers_neuroncount = delta[0]->get_row();
        if(this->backprop_helper->padded_delta[threadindex] == NULL)
        {
            this->backprop_helper->set_padded_delta_1d(delta, next_layers_neuroncount, (this->output_row-1)/2, (this->output_col-1)/2,
                                                       (this->output_row-1)/2, (this->output_col-1)/2, threadindex);
        }
        /*padded_delta = new Matrix* [next_layers_neuroncount];
        for(int i = 0; i < next_layers_neuroncount; i++)
        {
            padded_delta[i] = new Matrix;
            padded_delta[i][0].data[0][0] = delta[0][0].data[i][0];
            padded_delta[i][0] = padded_delta[i][0].zero_padd((this->output_row-1)/2,
                                                     (this->output_col-1)/2,
                                                     (this->output_row-1)/2,
                                                     (this->output_col-1)/2);
        }*/
        //Matrix kernel(this->output_row, this->output_col);
        for(int i = 0; i < this->map_count; i++)
        {
            this->layers_delta_helper[threadindex][i][0].zero();
            for(int j = 0; j < next_layers_neuroncount; j++)
            {
                this->get_2D_weights(j, i, this->backprop_helper->kernel[threadindex][0], next_layers_fmaps);
                calculate_delta_helper(this->backprop_helper->padded_delta[threadindex][j], layers_delta_helper[threadindex][i],
                                       this->backprop_helper->kernel[threadindex][0], this->backprop_helper->helper[threadindex][0]);
            }
        }
        this->backprop_helper->zero(threadindex);
        //delete_padded_delta(padded_delta, next_layers_neuroncount);
    }
    else if (this->next_layers_type == CONVOLUTIONAL)
    {
        if(this->backprop_helper->padded_delta[threadindex] == NULL)
        {
            this->backprop_helper->set_padded_delta_2d(delta, next_layers_fmapcount, next_layer, threadindex);
        }
        /*padded_delta = new Matrix* [next_layers_fmapcount];
        for(int i = 0; i < next_layers_fmapcount; i++)
        {
            padded_delta[i] = new Matrix;
            dilated = delta[i][0].dilate(static_cast<Convolutional*>(next_layer)->get_vertical_stride(), static_cast<Convolutional*>(next_layer)->get_horizontal_stride());
            padded_delta[i][0] = dilated.zero_padd((next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2);
        }*/
        for(int i = 0; i < this->map_count; i++)
        {
            this->layers_delta_helper[threadindex][i][0].zero();
            for(int j = 0; j < next_layers_fmapcount; j++)
            {
                calculate_delta_helper(this->backprop_helper->padded_delta[threadindex][j], this->layers_delta_helper[threadindex][i],
                                       next_layers_fmaps[j]->weights[i][0], this->backprop_helper->helper[threadindex][0]);
            }
        }
        this->backprop_helper->zero(threadindex);
        //delete_padded_delta(padded_delta, next_layers_fmapcount);
    }
    for(int i = 0; i < this->map_count; i++)
    {
        if(this->next_layers_type != POOLING)
        {
            this->layers_delta[threadindex][i][0] = hadamart_product(this->layers_delta_helper[threadindex][i][0], this->output_derivative[threadindex][i][0]);
        }
        else
        {
            this->layers_delta[threadindex][i][0] = hadamart_product(delta[i][0], this->output_derivative[threadindex][i][0]);
        }
        this->backprop_helper->dilated[threadindex] = this->layers_delta[threadindex][i][0].dilate(this->vertical_stride, this->horizontal_stride);
        for(int j = 0; j < this->fmap[i]->get_mapdepth(); j++)
        {
            cross_correlation(input[j][0], this->backprop_helper->dilated[threadindex], nabla[i]->weights[j][0], this->vertical_stride, this->horizontal_stride);
        }
        nabla[i]->biases[0][0].data[0][0] = this->layers_delta[threadindex][i][0].sum_over_elements();
    }
    return this->layers_delta[threadindex];
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *gradient)
{
    double sum;
    for(int i = 0; i < this->map_count; i++)
    {
        this->fmap[i]->biases[0]->data[0][0] -= learning_rate * this->fmap[i]->biases[0]->data[0][0];
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
    for(int channel_index = 0; channel_index < this->fmap[map_index]->get_mapdepth(); channel_index++)
    {
        convolution(input[channel_index][0], this->fmap[map_index]->weights[channel_index][0], convolved, this->vertical_stride, this->horizontal_stride);
        helper += convolved;
    }
    helper+=this->fmap[map_index]->biases[0][0].data[0][0];
}

void Convolutional::layers_output(Matrix **input, int threadindex)
{
    //Matrix convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    this->feedforward_helpter->helper[threadindex]->zero();
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        this->fulldepth_conv(this->feedforward_helpter->helper[threadindex][0], this->feedforward_helpter->convolved[threadindex][0], input, map_index);
        this->neuron.neuron(this->feedforward_helpter->helper[threadindex][0], this->outputs[threadindex][map_index][0]);
        //helper.zero();
        this->feedforward_helpter->helper[threadindex]->zero();
    }
}

Matrix** Convolutional::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

Matrix** Convolutional::derivate_layers_output(Matrix **input, int threadindex)
{
    //Matrix convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    this->feedforward_helpter->helper[threadindex]->zero();
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        this->fulldepth_conv(this->feedforward_helpter->helper[threadindex][0], this->feedforward_helpter->convolved[threadindex][0], input, map_index);
        this->neuron.neuron_derivative(this->feedforward_helpter->helper[threadindex][0], output_derivative[threadindex][map_index][0]);
        //helper.zero();
        this->feedforward_helpter->helper[threadindex]->zero();
    }
    return output_derivative[threadindex];
}

void Convolutional::flatten(int threadindex)
{
    int i = 0;
    int output_size = this->output_row * this->output_col;
    int output_size_in_bytes = output_size * sizeof(double);
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        memcpy(&(this->flattened_output[threadindex][0]->dv[map_index*output_size]), this->outputs[threadindex][map_index]->dv, output_size_in_bytes);
    }
}

void Convolutional::set_threadcount(int threadcnt)
{
    //***outputs, ***flattened_output, ****output_derivative;
    /*for(int i = 0; i < this->threadcount; i++)
    {
        for(int j = 0; j < this->map_count; j++)
        {
            delete this->outputs[i][j];
            //delete this->flattened_output[i][j];
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
    delete this->backprop_helper;*/
    this->destory_outputs_and_erros();

    this->threadcount = threadcnt;

    this->build_outputs_and_errors();

    /*this->feedforward_helpter = new conv_output_helper(threadcnt, this->output_row, this->output_col);
    this->backprop_helper = new conv_backprop_helper(threadcnt, this->output_row, this->output_col);

    this->outputs = new Matrix** [threadcnt];
    this->flattened_output = new Matrix** [threadcnt];
    this->output_derivative = new Matrix** [threadcnt];
    this->layers_delta = new Matrix** [threadcnt];
    this->layers_delta_helper = new Matrix** [threadcnt];
    for(int i = 0; i < threadcnt; i++)
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
    }*/
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
    if(this->next_layers_type == FULLY_CONNECTED)
    {
        this->flatten(threadindex);
        return this->flattened_output[threadindex];
    }
    else
        return this->outputs[threadindex];
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
