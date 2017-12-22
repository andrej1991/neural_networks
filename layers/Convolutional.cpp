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

inline void Convolutional::backpropagate(Matrice **input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error)
{
    ;
}

inline void Convolutional::layers_output(Matrice **input)
{
    Matrice convolved(this->output_row, this->output_col), helper(this->output_row, this->output_col);
    for(int map_index = 0; map_index < this->map_count; map_index++)
        {
            for(int channel_index = 0; channel_index < this->fmap[map_index]->mapdepth; channel_index++)
                {
                    convolution(input[channel_index][0], this->fmap[map_index]->weights[channel_index][0], convolved, this->stride);
                    helper += convolved;
                }
            helper+=this->fmap[map_index]->biases[0][0].data[0][0];
            this->outputs[map_index][0] = this->neuron.neuron(helper);
            helper.zero();
        }
}

inline Matrice Convolutional::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

inline Matrice Convolutional::derivate_layers_output(Matrice **input)
{
    ;
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, int prev_outputlen, Matrice *weights, Matrice *biases)
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

Matrice Convolutional::flatten_to_2D(Matrice &m)
{
    ;
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

inline Matrice* Convolutional::get_weights()
{
    ;
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
