#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "../matrix/matrix.h"

#define SIGMOID 0
#define RELU 1
#define LEAKY_RELU 2
#define TANH 3

class Neuron{
    int neuron_type;
    inline void sigmoid(Matrix &input, Matrix &output);
    inline void sigmoid_derivative(Matrix &input, Matrix &output);
    inline void tan_hip(Matrix &input, Matrix &output);
    inline void tan_hip_derivative(Matrix &input, Matrix &output);
    inline void relu(Matrix &input, Matrix &output);
    inline void leaky_relu(Matrix &input, Matrix &output);
    inline void relu_derivative(Matrix &input, Matrix &output);
    inline void leaky_relu_derivative(Matrix &input, Matrix &output);
    public:
    Neuron(int neuron_type = SIGMOID);
    void neuron(Matrix &inputs, Matrix &outputs);
    void neuron_derivative(Matrix &inputs, Matrix &outputs);
    void test();
};


#endif // NEURON_H_INCLUDED
