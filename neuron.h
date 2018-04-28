#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "matrix/matrix.h"

#define SIGMOID 0
#define RELU 1
#define LEAKY_RELU 2

class Neuron{
    int neuron_type;
    inline MatrixData sigmoid(MatrixData &input);
    inline MatrixData sigmoid_derivate(MatrixData &input);
    inline MatrixData relu(MatrixData &input);
    inline MatrixData leaky_relu(MatrixData &input);
    inline MatrixData relu_derivate(MatrixData &input);
    inline MatrixData leaky_relu_derivate(MatrixData &input);
    public:
    Neuron(int neuron_type = SIGMOID);
    MatrixData neuron(MatrixData &inputs);
    MatrixData neuron_derivate(MatrixData &inputs);
    void test();
};


#endif // NEURON_H_INCLUDED
