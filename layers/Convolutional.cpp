#include "layers.h"

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
    this->layer_type = CONVOLUTIONAL;
    this->fmap = new Feature_map* [map_count];
    this->outputs = new Matrix* [map_count];
    this->output_derivative = new Matrix* [map_count];
    this->layers_delta = new Matrix* [map_count];
    this->layers_delta_helper = new Matrix* [map_count];
    this->flattened_output = new Matrix* [1];
    this->flattened_output[0] = new Matrix(this->map_count * this->output_row * this->output_col, 1);
    for(int i = 0; i < map_count; i++)
    {
        fmap[i] = new Feature_map(this->kernel_row, this->kernel_col, input_channel_count);
        this->outputs[i] = new Matrix(this->output_row, this->output_col);
        this->output_derivative[i] = new Matrix(this->output_row, this->output_col);
        this->layers_delta[i] = new Matrix(this->output_row, this->output_col);
        this->layers_delta_helper[i] = new Matrix(this->output_row, this->output_col);
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
        delete output_derivative[i];
        delete layers_delta[i];
        delete layers_delta_helper[i];
    }
    delete[] fmap;
    delete[] outputs;
    delete[] output_derivative;
    delete[] layers_delta;
    delete[] layers_delta_helper;
}

void Convolutional::get_2D_weights(int neuron_id, int fmap_id, Matrix &kernel, Feature_map **next_layers_fmap)
{
    int kernelsize = kernel.get_row() * kernel.get_col();
    int starting_pos = kernelsize * fmap_id;
    int endpos = starting_pos + kernelsize;
    int index = starting_pos;
    for(int col = 0; col < kernel.get_col(); col++)
    {
        for(int row = 0; row < kernel.get_row(); row++)
        {
            kernel.data[row][col] = next_layers_fmap[0]->weights[0]->data[neuron_id][index];
            index++;
        }
    }
}

inline void calculate_delta_helper(Matrix *padded_delta, Matrix *delta_helper, Matrix &kernel, Matrix &helper)
{
    convolution(padded_delta[0],kernel, helper);
    //cross_correlation(padded_delta[0],kernel, helper);
    delta_helper[0] += helper;
}

inline void delete_padded_delta(Matrix **padded_delta, int limit)
{
    for(int i = 0; i < limit; i++)
    {
        delete padded_delta[i];
    }
    delete[] padded_delta;
}

inline Matrix** Convolutional::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **delta)
{
    Feature_map** next_layers_fmaps = next_layer->get_feature_maps();
    int next_layers_fmapcount = next_layer->get_mapcount();
    this->derivate_layers_output(input);
    Matrix **padded_delta;
    Matrix helper(this->output_row, this->output_col);
    Matrix dilated;
    if(this->next_layers_type != CONVOLUTIONAL)
    {
        int next_layers_neuroncount = delta[0]->get_row();
        padded_delta = new Matrix* [next_layers_neuroncount];
        for(int i = 0; i < next_layers_neuroncount; i++)
        {
            padded_delta[i] = new Matrix;
            padded_delta[i][0].data[0][0] = delta[0][0].data[i][0];
            padded_delta[i][0] = padded_delta[i][0].zero_padd((this->output_row-1)/2,
                                                     (this->output_col-1)/2,
                                                     (this->output_row-1)/2,
                                                     (this->output_col-1)/2);
        }
        Matrix kernel(this->output_row, this->output_col);
        for(int i = 0; i < this->map_count; i++)
        {
            this->layers_delta_helper[i][0].zero();
            for(int j = 0; j < next_layers_neuroncount; j++)
            {
                this->get_2D_weights(j, i, kernel, next_layers_fmaps);
                calculate_delta_helper(padded_delta[j], layers_delta_helper[i], kernel, helper);
            }
        }
        delete_padded_delta(padded_delta, next_layers_neuroncount);
    }
    else
    {
        padded_delta = new Matrix* [next_layers_fmapcount];
        for(int i = 0; i < next_layers_fmapcount; i++)
        {
            padded_delta[i] = new Matrix;
            dilated = delta[i][0].dilate(static_cast<Convolutional*>(next_layer)->get_vertical_stride(), static_cast<Convolutional*>(next_layer)->get_horizontal_stride());
            padded_delta[i][0] = dilated.zero_padd((next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2);
        }
        for(int i = 0; i < this->map_count; i++)
        {
            this->layers_delta_helper[i][0].zero();
            for(int j = 0; j < next_layers_fmapcount; j++)
            {
                calculate_delta_helper(padded_delta[j], this->layers_delta_helper[i], next_layers_fmaps[j]->weights[i][0], helper);
            }
        }
        delete_padded_delta(padded_delta, next_layers_fmapcount);
    }
    Matrix rotated_input;
    for(int i = 0; i < this->map_count; i++)
    {
        this->layers_delta[i][0] = hadamart_product(this->layers_delta_helper[i][0], this->output_derivative[i][0]);
        for(int j = 0; j < this->fmap[i]->get_mapdepth(); j++)
        {
            dilated = this->layers_delta[i][0].dilate(this->vertical_stride, this->horizontal_stride);
            convolution(input[j][0], dilated, nabla[i]->weights[j][0], this->vertical_stride, this->horizontal_stride);
            //rotated_input = input[j][0].rot180();
            //cross_correlation(rotated_input, dilated, nabla[i]->weights[j][0]);
        }
    }
    /*cout << input[0][0].get_row() << endl;
    cout << input[0][0].get_col() << endl;
    cout << dilated.get_row() << endl;
    cout << dilated.get_col() << endl;
    throw exception();*/
    return this->layers_delta;
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    for(int i = 0; i < this->map_count; i++)
    {
        for(int j = 0; j < this->fmap[i]->get_mapdepth(); j++)
        {
            for(int row = 0; row < this->kernel_row; row++)
            {
                for(int col = 0; col < this->kernel_col; col++)
                {
                    this->fmap[i]->weights[j]->data[row][col] =
                                    regularization_rate * this->fmap[i]->weights[j]->data[row][col] -
                                    learning_rate * layer->fmap[i]->weights[j]->data[row][col];
                }
            }
        }
    }
}

inline void Convolutional::fulldepth_conv(Matrix &helper, Matrix &convolved, Matrix **input, int map_index)
{
    for(int channel_index = 0; channel_index < this->fmap[map_index]->get_mapdepth(); channel_index++)
    {
        convolution(input[channel_index][0], this->fmap[map_index]->weights[channel_index][0], convolved, this->vertical_stride, this->horizontal_stride);
        helper += convolved;
    }
    helper+=this->fmap[map_index]->biases[0][0].data[0][0];
}

inline void Convolutional::layers_output(Matrix **input)
{
    Matrix convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        this->fulldepth_conv(helper, convolved, input, map_index);
        this->neuron.neuron(helper, this->outputs[map_index][0]);
        helper.zero();
    }
}

inline Matrix** Convolutional::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

inline Matrix** Convolutional::derivate_layers_output(Matrix **input)
{
    Matrix convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        this->fulldepth_conv(helper, convolved, input, map_index);
        this->neuron.neuron_derivative(helper, output_derivative[map_index][0]);
        helper.zero();
    }
    return output_derivative;
}

void Convolutional::flatten()
{
    int i = 0;
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        for(int col = 0; col < this->output_col; col++)
        {
            for(int row = 0; row < this->output_row; row++)
            {
                this->flattened_output[0]->data[i][0] = this->outputs[map_index]->data[row][col];
                i++;
            }
        }
    }
}


inline void Convolutional::remove_some_neurons(Matrix ***w_bckup, Matrix ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

inline void Convolutional::add_back_removed_neurons(Matrix **w_bckup, Matrix **b_bckup, int *layers_bckup, int **indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

void Convolutional::set_input(Matrix **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}

inline Matrix** Convolutional::get_output()
{
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
