#ifndef REINFORCE_H_INCLUDED
#define REINFORCE_H_INCLUDED

#include "../network.h"
#include "../SGD.h"

void print_action(Matrix &action);
void reinforcement_snake(Network &net, StochasticGradientDescent &learn, double learning_rate, double regularization_rate,
                                         int input_row, int input_col, double momentum, double denominator = 0.001, int input_channels = 1);


#endif // REINFORCE_H_INCLUDED
