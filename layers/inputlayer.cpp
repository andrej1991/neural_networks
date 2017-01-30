#include "layers.h"

InputLayer::InputLayer(int row, int col, int feature_depth, int neuron_type, Padding &p, short int next_layers_type):
    next_layers_type(next_layers_type), padd(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), feature_depth(feature_depth), output(row + p.top_padding + p.bottom_padding, col + p.left_padding + p.right_padding)
{
    this->outputlen = row;
    this->layer_type = INPUTLAYER;
    /*this->outputs = new Matrice* [feature_depth];
    for(int i = 0; i < feature_depth; i++)
        {
            outputs[i] = new Matrice(row + p.top_padding + p.bottom_padding, col + p.left_padding + p.right_padding);
        }*/
}

InputLayer::~InputLayer()
{
    ;
}

inline void InputLayer::layers_output(Matrice &input)
{
    ;
}

inline Matrice InputLayer::get_output_error(Matrice &input, Matrice &required_output, int costfunction_type)
{
    ;
}

inline Matrice InputLayer::derivate_layers_output(Matrice &input)
{
    ;
}

void InputLayer::update_weights_and_biasses(double learning_rate, double regularization_rate, int prev_outputlen, Matrice *weights, Matrice *biases)
{
    ;
}

inline void InputLayer::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void InputLayer::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void InputLayer::set_input(Matrice &input)
{
    ///TODO modify this function
    for (int i = 0; i < this->outputlen; i++)
        {
            this->output.data[i][0] = input.data[i][0];
        }
    double **debug = this->output.data;
    int j = 0;
}

inline void InputLayer::backpropagate(Matrice &input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error)
{
    ;
}

inline Matrice* InputLayer::get_output()
{
    return &(this->output);
}

inline Matrice* InputLayer::get_weights()
{
    ;
}

inline short InputLayer::get_layer_type()
{
    return this->layer_type;
}

inline int InputLayer::get_outputlen()
{
    return this->outputlen;
}

void InputLayer::set_weights(Matrice *w)
{
    ;
}

void InputLayer::set_biases(Matrice *b)
{
    ;
}
