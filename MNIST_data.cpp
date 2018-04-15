#include <iostream>
#include <fstream>
#include "MNIST_data.h"

using namespace std;


MNIST_data::MNIST_data(int input_vector_row, int input_vector_col, int output_vector_size, int feature_depth):
    input_vector_row(input_vector_row), input_vector_col(input_vector_col), output_vector_size(output_vector_size), feature_depth(feature_depth), required_output(output_vector_size, 1)
{
    this->input = new Matrice* [feature_depth];
    for(int i = 0; i < feature_depth; i++)
        {
            input[i] = new Matrice(input_vector_row, input_vector_col);
        }
}

MNIST_data::~MNIST_data()
{
    for(int i = 0; i < this->feature_depth; i++)
        {
            delete[] this->input[i];
        }
    delete[] this->input;
}

void MNIST_data::load_data(std::ifstream &input, std::ifstream &required_output)
{
    double *inp = new double [this->input_vector_row * this->input_vector_col];
    double *req = new double [this->output_vector_size];
    for(int k = 0; k < this->feature_depth; k++)
        {
            input.read((char*)inp, this->input_vector_row * this->input_vector_col * sizeof(double));
            for(int i = 0; i < this->input_vector_row; i++)
                {
                    for(int j = 0; j < this->input_vector_col; j++)
                        {
                            (this->input[k][0])[i][j] = inp[i * this->input_vector_col + j];
                        }
                }
        }
    required_output.read((char*)req, this->output_vector_size  * sizeof(double));
    for(int i = 0; i < this->output_vector_size; i++)
        {
            this->required_output[i][0] = req[i];
        }
    delete[] inp;
    delete[] req;
}
