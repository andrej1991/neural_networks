#include "layers.h"

InputLayer::InputLayer(int row, int col, int input_channel_count, int neuron_type, Padding &p, short int next_layers_type):
    next_layers_type(next_layers_type), padd(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), input_channel_count(input_channel_count),
    row(row), col(col)
{
    this->outputlen = row;
    this->layer_type = INPUTLAYER;
    this->outputs = new Matrix* [input_channel_count];
    for(int i = 0; i < input_channel_count; i++)
        {
            outputs[i] = new Matrix(row + p.top_padding + p.bottom_padding, col + p.left_padding + p.right_padding);
        }
}

InputLayer::~InputLayer()
{
    for(int i = 0; i < this->input_channel_count; i++)
        {
            delete this->outputs[i];
        }
    delete[] this->outputs;
}

inline void InputLayer::layers_output(Matrix **input)
{
    this->set_input(input);
}

inline Matrix** InputLayer::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type)
{
    ;
}

inline Matrix** InputLayer::derivate_layers_output(Matrix **input)
{
    ;
}

void InputLayer::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    ;
}

void InputLayer::set_input(Matrix **input)
{
    ///TODO modify this function to work with FC layer and convolutional layer
    if(this->next_layers_type == FULLY_CONNECTED)
        {
            for (int i = 0; i < this->input_channel_count; i++)
                {
                    this->outputs[i][0] = input[i][0];
                }
        }
    else if(this->next_layers_type == CONVOLUTIONAL)
        {
            for(int l = 0; l < this->input_channel_count; l++)
                {
                    //int debug1 = this->outputs[l]->get_col();
                    //int debug2 = this->outputs[l]->get_row();
                    for(int i = 0; i < this->row; i++)
                        {
                            for(int j = 0; j < this->col; j++)
                                {
                                    this->outputs[l][0].data[i][j] = input[l][0].data[i][j];
                                }
                        }
                }
        }
}

inline Matrix** InputLayer::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error)
{
    ;
}

inline Matrix** InputLayer::get_output()
{
    return this->outputs;
}

inline Feature_map** InputLayer::get_feature_maps()
{
    ;
}

inline short InputLayer::get_layer_type()
{
    return this->layer_type;
}

inline int InputLayer::get_output_row()
{
    return this->row;
}

inline int InputLayer::get_output_len()
{
    return this->row;
}

inline int InputLayer::get_output_col()
{
    return this->col;
}

int InputLayer::get_weights_row()
{
    return this->row;
}

int InputLayer::get_weights_col()
{
    return this->col;
}

int InputLayer::get_mapcount()
{
    return this->input_channel_count;
}

int InputLayer::get_mapdepth()
{
    1;
}

void InputLayer::store(std::ofstream &params)
{
    ;
}

void InputLayer::load(std::ifstream &params)
{
    ;
}
