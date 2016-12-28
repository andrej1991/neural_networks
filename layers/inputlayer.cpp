#include "layers.h"

InputLayer::InputLayer()
{
    ;
}

InputLayer::~InputLayer()
{
    ;
}

inline void InputLayer::layers_output(double **input, int layer)
{
    ;
}

inline Matrice InputLayer::get_delta(double **output, double **required_output)
{
    ;
}

inline Matrice InputLayer::derivate_layers_output(double **input)
{
    ;
}

void InputLayer::update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases)
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

void InputLayer::set_input(double **input, int input_len)
{
    ;
}
