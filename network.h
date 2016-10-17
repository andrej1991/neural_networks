#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED
#include "neuron.h"
#include "MNIST_data.h"
#include "matrice.h"


class Network{
    int total_layers_num, layers_num;
    int *layers;
    Matrice **biases, **outputs, **weights;
    void initialize_biases();
    void initialize_weights();
    inline void layers_output(double **input, int layer);
    inline void backpropagate(MNIST_data *training_data, Matrice **nabla_b, Matrice **nabla_w);
    inline Matrice cost_derivate(double **output, double **required_output);
    inline Matrice derivate_layers_output(int layer, double **input);
    void update_weights_and_biasses(MNIST_data **training_data, int training_data_len, double learning_rate);
    inline void feedforward(double **input);
    Neuron neuron;
    public:
    void stochastic_gradient_descent(MNIST_data **training_data, int epochs, int epoch_len, double learning_rate, int trainingdata_len = 50000);
    Network(int layers_num, int *layers);
    ~Network();
    void test(MNIST_data **d);
    Matrice get_output(double **input);
};

#endif // NETWORK_H_INCLUDED
