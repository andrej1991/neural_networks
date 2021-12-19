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

void MNIST_data::load_bmp(const char *path)
{
    ifstream inp;
    short int signature, bits_per_pixel;
    int data_offset, width, height, compression, img_size;
    try
    {
        inp.open(path, ios::in|ios::binary);
        inp.read((char*) &signature, 2);
        inp.seekg(0xA);
        inp.read((char*) &data_offset, 4);
        inp.seekg(0x12);
        inp.read((char*) &width, 4);
        inp.read((char*) &height, 4);
        inp.seekg(0x1C);
        inp.read((char*) &bits_per_pixel, 2);
        inp.read((char*) &compression, 4);
        inp.read((char*) &img_size, 4);
        if(signature == 19778)
        {
            throw invalid_argument("The file is not a BMP file!\n");
        }
        char temp_inp[4 * this->input_vector_row * this->input_vector_col];
        inp.seekg(data_offset);
        unsigned char tmp;
        inp.read(temp_inp, this->input_vector_row * this->input_vector_col * 4);
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
    catch( exception &e)
    {
        inp.close();
        throw e;
    }
    if(inp.is_open())
    {
        inp.close();
    }
}

