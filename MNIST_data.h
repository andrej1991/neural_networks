#ifndef MNIST_DATA_H_INCLUDED
#define MNIST_DATA_H_INCLUDED
#include <fstream>

class MNIST_data{
    public:
    double **input;
    double **required_output;
    int input_vector_size, output_vector_size;
    MNIST_data(int input_vector_size, int output_vector_size);
    ~MNIST_data();
    void load_data(std::ifstream &input, std::ifstream &required_output);

};

#endif // MNIST_DATA_H_INCLUDED
