#include <iostream>
#include <fstream>
#include "MNIST_data.h"

using namespace std;


MNIST_data::MNIST_data(int input_vector_row, int input_vector_col, int output_vector_size, int feature_depth):
    input_vector_row(input_vector_row), input_vector_col(input_vector_col), output_vector_size(output_vector_size), feature_depth(feature_depth), required_output(output_vector_size, 1)
{
    this->input = new Matrix* [feature_depth];
    for(int i = 0; i < feature_depth; i++)
    {
        input[i] = new Matrix(input_vector_row, input_vector_col);
    }
}

MNIST_data::~MNIST_data()
{
    for(int i = 0; i < this->feature_depth; i++)
    {
        delete this->input[i];
    }
    delete[] this->input;
}

void MNIST_data::load_data(std::ifstream &input, std::ifstream &required_output)
{
    for(int k = 0; k < this->feature_depth; k++)
    {
        input.read((char*)this->input[k]->dv, this->input_vector_row * this->input_vector_col * sizeof(double));
    }
    required_output.read((char*)this->required_output.dv, this->output_vector_size  * sizeof(double));
}

void MNIST_data::load_bmp(std::ifstream &in)
{
    char temp_inp[4 * this->input_vector_row * this->input_vector_col];
    in.seekg(0x7a);
    unsigned char tmp;
    in.read(temp_inp, this->input_vector_row * this->input_vector_col * 4);
    for(int i = 0; i < this->input_vector_row; i++)
    {
        for(int j = 0; j < this->input_vector_col; j++)
        {
            tmp = temp_inp[4*(i * this->input_vector_col + j)];
            this->input[0]->data[i][j] = tmp;
            tmp = temp_inp[4*(i * this->input_vector_col + j) + 1];
            this->input[1]->data[i][j] = tmp;
            tmp = temp_inp[4*(i * this->input_vector_col + j) + 2];
            this->input[2]->data[i][j] = tmp;
        }
    }
}

