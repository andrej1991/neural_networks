#include "layers.h"

InputLayer::InputLayer(int row, int col, int neuron_type):
    output(row, col)
{
    this->outputlen = row;
    this->layer_type = INPUTLAYER;
}

InputLayer::~InputLayer()
{
    ;
}

inline void InputLayer::layers_output(Matrice &input)
{
    ;
}

inline Matrice InputLayer::get_output_error(Matrice &input, double **required_output, int costfunction_type)
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

void InputLayer::set_input(double **input)
{
    for (int i = 0; i < this->outputlen; i++)
        {
            this->output.data[i][0] = input[i][0];
        }
    double **debug = this->output.data;
    int j = 0;
}

inline void InputLayer::backpropagate(Matrice &input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error)
{
    ;
}

inline Matrice& InputLayer::get_output()
{
    return this->output;
}

inline Matrice& InputLayer::get_weights()
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

inline int InputLayer::get_neuron_count()
{
    ;
}
