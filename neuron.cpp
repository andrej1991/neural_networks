#include "neuron.h"
#include <math.h>
#include <iostream>

Neuron::Neuron(int neuron_type) : neuron_type(neuron_type) {}

inline Matrice Neuron::sigmoid(Matrice &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    Matrice ret(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    ret.data[i][j] = 1 / (1 + exp(-1 * inputs.data[i][j]));
                }
        }
    return ret;
}

inline Matrice Neuron::sigmoid_derivate(Matrice &inputs)
{
    Matrice s = sigmoid(inputs);
    int row = inputs.get_row();
    int col = inputs.get_col();
     for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    s.data[i][j] = s.data[i][j] * (1 - s.data[i][j]);
                }
        }
    return s;
}

inline Matrice Neuron::relu(Matrice &inputs)
{
    ;
}
inline Matrice Neuron::relu_derivate(Matrice &inputs)
{
    ;
}

Matrice Neuron::neuron(Matrice &inputs)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid(inputs);
    case RELU:
        return this->relu(inputs);
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}

Matrice Neuron::neuron_derivate(Matrice &inputs)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid_derivate(inputs);
    case RELU:
        return this->relu_derivate(inputs);
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}



