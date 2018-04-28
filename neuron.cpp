#include "neuron.h"
#include <math.h>
#include <iostream>

Neuron::Neuron(int neuron_type) : neuron_type(neuron_type) {}

/*inline Matrice Neuron::sigmoid(Matrice &inputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    Matrice ret(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    ret[i][j] = 1 / (1 + exp(-1 * inputs[i][j]));
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
                    s[i][j] = s[i][j] * (1 - s[i][j]);
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
                    if(inputs[i][j] > 0)
                        output[i][j] = inputs[i][j];
                    else
                        output[i][j] = 0;
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
                    if(inputs[i][j] > 0)
                        output[i][j] = 1;
                    else
                        output[i][j] = 0;
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
                    if(inputs[i][j] > 0)
                        output[i][j] = inputs[i][j];
                    else
                        output[i][j] = 0.001*inputs[i][j];
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
                    if(inputs[i][j] > 0)
                        output[i][j] = 1;
                    else
                        output[i][j] = 0.001;
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
}*/



