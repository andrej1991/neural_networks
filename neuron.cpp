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
    int row = inputs.get_row();
    int col = inputs.get_col();
    Matrice output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        output.data[i][j] = inputs.data[i][j];
                    else
                        output.data[i][j] = 0;
                }
        }
    return output;
}
inline Matrice Neuron::relu_derivate(Matrice &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    Matrice output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        output.data[i][j] = 1;
                    else
                        output.data[i][j] = 0;
                }
        }
    return output;
}

inline Matrice Neuron::leaky_relu(Matrice &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    Matrice output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        output.data[i][j] = inputs.data[i][j];
                    else
                        output.data[i][j] = 0.001*inputs.data[i][j];
                }
        }
    return output;
}
inline Matrice Neuron::leaky_relu_derivate(Matrice &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    Matrice output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        output.data[i][j] = 1;
                    else
                        output.data[i][j] = 0.001;
                }
        }
    return output;
}

Matrice Neuron::neuron(Matrice &inputs)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid(inputs);
    case RELU:
        return this->relu(inputs);
    case LEAKY_RELU:
        return this->leaky_relu(inputs);
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
    case LEAKY_RELU:
        return this->leaky_relu_derivate(inputs);
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}



