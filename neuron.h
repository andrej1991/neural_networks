#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "matrice.h"

#define SIGMOID 0
#define RELU 1
#define LEAKY_RELU 2

class Neuron{
    int neuron_type;
    inline void sigmoid(Matrice &input, Matrice &output);
    inline void sigmoid_derivative(Matrice &input, Matrice &output);
    inline void relu(Matrice &input, Matrice &output);
    inline void leaky_relu(Matrice &input, Matrice &output);
    inline void relu_derivative(Matrice &input, Matrice &output);
    inline void leaky_relu_derivative(Matrice &input, Matrice &output);
    public:
    Neuron(int neuron_type = SIGMOID);
    void neuron(Matrice &inputs, Matrice &outputs);
    void neuron_derivative(Matrice &inputs, Matrice &outputs);
    void test();
};


#endif // NEURON_H_INCLUDED
