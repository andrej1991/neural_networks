#include "layers.h"

InputLayer::InputLayer(int row, int col, int input_channel_count, int neuron_type, Padding &p, short int next_layers_type):
    next_layers_type(next_layers_type), padd(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), input_channel_count(input_channel_count),
    row(row), col(col)
{
    this->threadcount = 1;
    this->outputlen = row;
    this->layer_type = INPUTLAYER;
    this->outputs = new Matrix** [1];
    this->outputs[0] = new Matrix* [input_channel_count];
    for(int i = 0; i < input_channel_count; i++)
    {
        outputs[0][i] = new Matrix(row + p.top_padding + p.bottom_padding, col + p.left_padding + p.right_padding);
    }
}

InputLayer::~InputLayer()
{
    for(int i = 0; i < this->input_channel_count; i++)
    {
        delete this->outputs[0][i];
    }
    delete[] this->outputs;
}

void InputLayer::layers_output(Matrix **input, int threadindex)
{
    this->set_input(input, threadindex);
}

Matrix* InputLayer::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    ;
}

Matrix** InputLayer::derivate_layers_output(Matrix **input, int threadindex)
{
    ;
}

void InputLayer::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    ;
}

void InputLayer::set_input(Matrix **input, int threadindex)
{
    //if(this->next_layers_type == FULLY_CONNECTED)
    //{
        for (int i = 0; i < this->input_channel_count; i++)
        {
            this->outputs[threadindex][i][0] = input[i][0];
        }
    /*}
    else if(this->next_layers_type == CONVOLUTIONAL)
    {
        for(int l = 0; l < this->input_channel_count; l++)
        {
            this->outputs[threadindex][l][0] = input[l][0] * (1.0/256);
        }
    }*/
}

inline int InputLayer::get_threadcount()
{
    return this->threadcount;
}

void InputLayer::set_threadcount(int threadcnt, vector<Matrix***> inputs_)
{
    for(int i = 0; i < this->threadcount; i++)
    {
        for(int j = 0; j < this->input_channel_count; j++)
        {
            delete this->outputs[i][j];
        }
    delete[] this->outputs[i];
    }
    delete[] this->outputs;
    this->threadcount = threadcnt;
    this->outputs = new Matrix** [threadcnt];
    for(int i = 0; i < threadcnt; i++)
    {
        this->outputs[i] = new Matrix* [input_channel_count];
        for(int j = 0; j < input_channel_count; j++)
        {
            outputs[i][j] = new Matrix(row + padd.top_padding + padd.bottom_padding, col + padd.left_padding + padd.right_padding);
        }
    }
}

Matrix** InputLayer::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***next_layers_error, int threadindex)
{
    ;
}

inline Matrix** InputLayer::get_output(int threadindex)
{
    return this->outputs[threadindex];
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
    return this->row * this->col;
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
    return 1;
}

void InputLayer::store(std::ofstream &params)
{
    ;
}

void InputLayer::load(std::ifstream &params)
{
    ;
}
