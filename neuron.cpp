#include "neuron.h"
#include <math.h>

double Neuron::sigmoid(double *weights, double **inputs, double bias, int inputs_len)
{
    double weighted_input = 0;
    for(int i = 0; i < inputs_len; i++)
        {
            weighted_input += weights[i] * inputs[i][0];
        }
    weighted_input += bias;
    return (1 / (1 + exp(-1 * weighted_input)));
}

double Neuron::sigmoid_derivate(double *weights, double **inputs, double bias, int inputs_len)
{
    double s = sigmoid(weights, inputs, bias, inputs_len);
    return (s * (1 - s));
}

void Neuron::test()
{
    ;
}

