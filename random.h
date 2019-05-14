#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED
#include <fstream>
#include "MNIST_data.h"

int random(int start, int end, std::ifstream &rnd);
int shuffle(MNIST_data **array, int len, std::ifstream &rnd);

#endif // RANDOM_H_INCLUDED
