#ifndef LAYERS_H_INCLUDED
#define LAYERS_H_INCLUDED

#include "neuron.h"
#include "MNIST_data.h"
#include "matrice.h"

class Layer{
    public:
    Neuron neuron;
    Layer();
    virtual ~Layer();
    virtual inline void layers_output(double **input, int layer);
    virtual inline Matrice get_delta(double **output, double **required_output);
    virtual inline Matrice derivate_layers_output(int layer, double **input);
    virtual void update_weights_and_biasses(MNIST_data **training_data, int training_data_len, int total_trainingdata_len, double learning_rate, double regularization_rate);
    virtual inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    virtual inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
};

#endif // LAYERS_H_INCLUDED
