#include "layers.h"

FullyConnected::FullyConnected()
{
    ;
}

FullyConnected::~FullyConnected()
{
    ;
}

inline void FullyConnected::layers_output(double **input, int layer)
{
    ;
}

inline Matrice FullyConnected::get_delta(double **output, double **required_output)
{
    ;
}

inline Matrice FullyConnected::derivate_layers_output(int layer, double **input)
{
    ;
}

void FullyConnected::update_weights_and_biasses(MNIST_data **training_data, int training_data_len, int total_trainingdata_len, double learning_rate, double regularization_rate)
{
    ;
}

inline void FullyConnected::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void FullyConnected::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

