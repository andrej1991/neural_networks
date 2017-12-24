#include "layers.h"

Convolutional::Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int stride):
                    input_row(input_row), input_col(input_col), kernel_row(kern_row), kernel_col(kern_col), map_count(map_count), stride(stride), next_layers_type(next_layers_type),
                    pad(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), neuron(neuron_type), neuron_type(neuron_type)
{
    if(stride != 1)
        {
            std::cerr << "counting with stride different than 1 is not implemented yet!";
            throw exception();
        }
    this->output_row = input_row - kern_row + 1;
    this->output_col = input_col - kern_col + 1;
    this->layer_type = CONVOLUTIONAL;
    this->fmap = new Feature_map* [map_count];
    this->outputs = new Matrice* [map_count];
    for(int i = 0; i < map_count; i++)
        {
            fmap[i] = new Feature_map(this->kernel_row, this->kernel_col, input_channel_count);
            this->outputs[i] = new Matrice(this->output_row, this->output_col);
        }
}

Convolutional::~Convolutional()
{
    ;
}

inline void Convolutional::backpropagate(Matrice **input, Feature_map** next_layers_fmaps, Feature_map** nabla, Matrice **delta, int next_layers_fmapcount)
{
    /*Matrice multiplied, **output_derivate;
    output_derivate = this->derivate_layers_output(input);
    multiplied = (next_layers_fmaps[0][0].weights[0]->transpose()) * delta[0][0];
    delta[0][0] = hadamart_product(multiplied, **output_derivate);
    nabla[0][0].biases[0][0] = delta[0][0];
    nabla[0][0].weights[0][0] = delta[0][0] * input[0][0].transpose();
    delete[] output_derivate;*/
    if(this->next_layers_type == FULLY_CONNECTED)
        {
            Matrice multiplied, **output_derivate;
            Matrice **delta_helper = new Matrice* [this->map_count];
            /*for(int j = 0; j < this->map_count; j++)
                {
                    delta_helper[j] = new Matrice(this->output_row, this->output_col);
                }*/
            output_derivate = this->derivate_layers_output(input);
            multiplied = (next_layers_fmaps[0][0].weights[0]->transpose()) * delta[0][0];
            for(int j = 0; j < this->map_count; j++)
                {
                    delta_helper[j] = new Matrice(this->output_row, this->output_col);
                    delta_helper[j][0] = hadamart_product(multiplied, **output_derivate);
                    nabla[0][0].biases[0][0] = delta_helper[j][0];
                    nabla[0][0].weights[0][0] = delta[0][0] * input[0][0].transpose();
                }
        }
}

inline void Convolutional::fulldepth_conv(Matrice &helper, Matrice &convolved)
{
    for(int channel_index = 0; channel_index < this->fmap[map_index]->get_mapdepth(); channel_index++)
        {
            convolution(input[channel_index][0], this->fmap[map_index]->weights[channel_index][0], convolved, this->stride);
            helper += convolved;
        }
    helper+=this->fmap[map_index]->biases[0][0].data[0][0];
}

inline void Convolutional::layers_output(Matrice **input)
{
    Matrice convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            this->fulldepth_conv(helper, convolved);
            this->outputs[map_index][0] = this->neuron.neuron(helper);
            helper.zero();
        }
}

inline Matrice Convolutional::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

inline Matrice** Convolutional::derivate_layers_output(Matrice **input)
{
    Matrice convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    Matrice **ret = new Matrice* [this->map_count];
    for(int i = 0; i < this->map_count; i++)
        {
            ret[i] = new Matrice(this->output_row, this->output_col);
        }
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            this->fulldepth_conv(helper, convolved);
            this->ret[map_index][0] = this->neuron.neuron_derivate(helper);
            helper.zero();
        }
    return ret;
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    ;
}

void Convolutional::flatten()
{
    ///TODO rewrite this if the feature maps can have different kernel size;
    this->flattened_output = new Matrice* [1];
    this->flattened_output[0] = new Matrice(this->map_count * this->output_row * this->output_col, 1);
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

Matrice** Convolutional::flatten_to_2D(Matrice &m)
{
    ///TODO rewrite this if the feature maps can have different kernel size;
    Matrice **ret = new Matrice* [this->map_count];
    for(int j = 0; j < this->map_count; j++)
        {
            ret[j] = new Matrice(this->output_row, this->output_col);
        }
    int i = 0;
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            for(int col = 0; col < this->output_col; col++)
                {
                    for(int row = 0; row < this->output_row; row++)
                        {
                            ret[map_index]->data[row][col] = this->flattened_output[0]->data[i][0];
                            i++;
                        }
                }
        }
    return ret;
}

inline void Convolutional::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

inline void Convolutional::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

void Convolutional::set_input(Matrice **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}

inline Matrice** Convolutional::get_output()
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

inline int Convolutional::get_outputlen()
{
    return (this->map_count * this->output_row * this->output_col);
}

void Convolutional::set_weights(Matrice *w)
{
    ;
}

void Convolutional::set_biases(Matrice *b)
{
    ;
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
