#ifndef MNIST_DATA_H_INCLUDED
#define MNIST_DATA_H_INCLUDED
#include <fstream>
#include "matrice.h"

class MNIST_data{
    public:
    Matrice **input;
    Matrice required_output;
    int input_vector_row, input_vector_col, feature_depth, output_vector_size;
    MNIST_data(int input_row, int input_col, int output_vector_size, int feature_depth);
    ~MNIST_data();
    void load_data(std::ifstream &input, std::ifstream &required_output);

};

#endif // MNIST_DATA_H_INCLUDED
