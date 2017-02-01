#include "layers.h"

Convolutional::Convolutional(int input_row, int input_col, int input_layer_count, int row, int col, int layer_count, int neuron_type, int stride):
                    input_row(input_row), input_col(input_col), row(row), col(col), layer_count(layer_count), stride(stride)
{
    if(stride != 1)
        {
            std::cerr << "counting with stride different than 1 is not implemented yet!";
            throw exception();
        }
    Matrice *helper;
    this->layer_type = CONVOLUTIONAL;
    int output_row = input_row - row + 1;
    int output_col = input_col - col + 1;
    helper = new Matrice(output_row * output_col * layer_count, 1);
    this->output = *helper;
    delete helper;
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
    ;
}

inline Matrice Convolutional::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    ;
}

inline Matrice Convolutional::derivate_layers_output(Matrice **input)
{
    ;
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, int prev_outputlen, Matrice *weights, Matrice *biases)
{
    ;
}

inline void Convolutional::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void Convolutional::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void Convolutional::set_input(Matrice **input)
{
    ;
}

inline Matrice** Convolutional::get_output()
{
    ;
}

inline Matrice* Convolutional::get_weights()
{
    ;
}

inline short Convolutional::get_layer_type()
{
    ;
}

inline int Convolutional::get_outputlen()
{
    ;
}

void Convolutional::set_weights(Matrice *w)
{
    ;
}

void Convolutional::set_biases(Matrice *b)
{
    ;
}
