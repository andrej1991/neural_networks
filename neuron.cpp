#include "neuron.h"
#include <math.h>
#include <iostream>

#define SIG(param)   (1 / (1 + exp(-1 * param)))

Neuron::Neuron(int neuron_type) : neuron_type(neuron_type) {}

inline void Neuron::sigmoid(Matrix &inputs, Matrix &outputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    //Matrix ret(row, col);
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            //outputs.data[i][j] = 1 / (1 + exp(-1 * inputs.data[i][j]));
            outputs.data[i][j] = SIG(inputs.data[i][j]);
        }
    }
    //return ret;
}

inline void Neuron::sigmoid_derivative(Matrix &inputs, Matrix &outputs)
{
    //Matrix s = sigmoid(inputs);
    int row = inputs.get_row();
    int col = inputs.get_col();
    double s;
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            s = SIG(inputs.data[i][j]);
            //s.data[i][j] = s.data[i][j] * (1 - s.data[i][j]);
            outputs.data[i][j] = s * (1 - s);
        }
    }
    //return s;
}

inline void Neuron::relu(Matrix &inputs, Matrix &outputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    //Matrix output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = inputs.data[i][j];
                    else
                        outputs.data[i][j] = 0;
                }
        }
    //return output;
}
inline void Neuron::relu_derivative(Matrix &inputs, Matrix &outputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    //Matrix output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = 1;
                    else
                        outputs.data[i][j] = 0;
                }
        }
    //return output;
}

inline void Neuron::leaky_relu(Matrix &inputs, Matrix &outputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    //Matrix output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = inputs.data[i][j];
                    else
                        outputs.data[i][j] = 0.001*inputs.data[i][j];
                }
        }
    //return output;
}
inline void Neuron::leaky_relu_derivative(Matrix &inputs, Matrix &outputs)
{
    int row = inputs.get_row();
    int col = inputs.get_col();
    //Matrix output(row, col);
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
                {
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = 1;
                    else
                        outputs.data[i][j] = 0.001;
                }
        }
    //return output;
}

void Neuron::neuron(Matrix &inputs, Matrix &outputs)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        this->sigmoid(inputs, outputs);
        return;
    case RELU:
        this->relu(inputs, outputs);
        return;
    case LEAKY_RELU:
        this->leaky_relu(inputs, outputs);
        return;
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}

void Neuron::neuron_derivative(Matrix &inputs, Matrix &outputs)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        this->sigmoid_derivative(inputs, outputs);
        return;
    case RELU:
        this->relu_derivative(inputs, outputs);
        return;
    case LEAKY_RELU:
        this->leaky_relu_derivative(inputs, outputs);
        return;
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}



