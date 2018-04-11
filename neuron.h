#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "matrice.h"

#define SIGMOID 0
#define RELU 1
#define LEAKY_RELU 2

class Neuron{
    int neuron_type;
    inline Matrice sigmoid(Matrice &input);
    inline Matrice sigmoid_derivate(Matrice &input);
    inline Matrice relu(Matrice &input);
    inline Matrice leaky_relu(Matrice &input);
    inline Matrice relu_derivate(Matrice &input);
    inline Matrice leaky_relu_derivate(Matrice &input);
    public:
    Neuron(int neuron_type = SIGMOID);
    Matrice neuron(Matrice &inputs);
    Matrice neuron_derivate(Matrice &inputs);
    void test();
};


#endif // NEURON_H_INCLUDED
