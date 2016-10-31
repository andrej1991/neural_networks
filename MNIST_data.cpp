#include <iostream>
#include <fstream>
#include "MNIST_data.h"

using namespace std;


MNIST_data::MNIST_data(int input_vector_size, int output_vector_size)
    {
        this->input = new double* [input_vector_size];
        this->required_output = new double* [output_vector_size];
        this->input_vector_size = input_vector_size;
        this->output_vector_size = output_vector_size;
    }
MNIST_data::~MNIST_data()
    {
        for(int i = 0; i < this->input_vector_size; i++)
            {
                delete[] this->input[i];
            }
        for(int i = 0; i < this->output_vector_size; i++)
            {
                delete[] this->required_output[i];
            }
        delete[] this->input;
        delete[] this->required_output;
    }
void MNIST_data::load_data(std::ifstream &input, std::ifstream &required_output)
    {
        double *inp = new double [this->input_vector_size];
        double *req = new double [this->output_vector_size];
        input.read((char*)inp, this->input_vector_size * sizeof(double));
        required_output.read((char*)req, this->output_vector_size  * sizeof(double));
        for(int i = 0; i < this->input_vector_size; i++)
            {
                this->input[i] = new double [1];
                this->input[i][0] = inp[i];
            }
        for(int i = 0; i < this->output_vector_size; i++)
            {
                this->required_output[i] = new double [1];
                this->required_output[i][0] = req[i];
            }
        delete[] inp;
        delete[] req;
    }
