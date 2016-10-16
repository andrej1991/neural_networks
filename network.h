#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED
#include "neuron.h"
#include "MNIST_data.h"
#include "matrice.h"


class Network{
    int layers_num;
    int *layers;
    Matrice **biases, **outputs, **weights;
    void initialize_biases();
    void initialize_weights();
    inline void layers_output(double **input, int layer);
    inline void backpropagate(MNIST_data *training_data, Matrice **nabla_b, Matrice **nabla_w);
    inline Matrice cost_derivate(double **output, double **required_output);
    inline Matrice derivate_layers_output(int layer, double **input);
    inline int get_inputlen(int layer);
    void update_weights_and_biasses(MNIST_data **training_data, int training_data_len, double learning_rate);
    inline void feedforward(double **input);
    Neuron neuron;
    public:
    Network(int layers_num, int *layers);
    ~Network();
    void test(MNIST_data **d);
    Matrice get_output(double **input);
};

#endif // NETWORK_H_INCLUDED
