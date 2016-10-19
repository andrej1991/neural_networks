#include "neuron.h"
#include <math.h>
#include <iostream>

Neuron::Neuron(int neuron_type) : neuron_type(neuron_type) {}

inline double Neuron::sigmoid(double *weights, double **inputs, double bias, int inputs_len)
{
    double weighted_input = 0;
    for(int i = 0; i < inputs_len; i++)
        {
            weighted_input += weights[i] * inputs[i][0];
        }
    weighted_input += bias;
    return (1 / (1 + exp(-1 * weighted_input)));
}

inline double Neuron::sigmoid_derivate(double *weights, double **inputs, double bias, int inputs_len)
{
    double s = sigmoid(weights, inputs, bias, inputs_len);
    return (s * (1 - s));
}

inline double Neuron::linear(double *weights, double **inputs, double bias, int inputs_len)
{
    double weighted_input = 0;
    for(int i = 0; i < inputs_len; i++)
        {
            weighted_input += weights[i] * inputs[i][0];
        }
    weighted_input += bias;
    return weighted_input;
}
inline double Neuron::linear_derivate(double *weights, double **inputs, double bias, int inputs_len)
{
    double weighted_input = 0;
    for(int i = 0; i < inputs_len; i++)
        {
            weighted_input += weights[i];
        }
    return weighted_input;
}

double Neuron::neuron(double *weights, double **inputs, double bias, int inputs_len)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid(weights, inputs, bias, inputs_len);
    case LINEAR:
        return this->linear(weights, inputs, bias, inputs_len);
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}

double Neuron::neuron_derivate(double *weights, double **inputs, double bias, int inputs_len)
{
    switch(this->neuron_type)
    {
    case SIGMOID:
        return this->sigmoid_derivate(weights, inputs, bias, inputs_len);
    case LINEAR:
        return this->linear_derivate(weights, inputs, bias, inputs_len);
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}



