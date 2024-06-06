#include <iostream>
#include <fstream>
#include "data_loader.h"

using namespace std;


Data_Loader::Data_Loader(int input_vector_row, int input_vector_col, int output_vector_size, int feature_depth):
    input_vector_row(input_vector_row), input_vector_col(input_vector_col), output_vector_size(output_vector_size), feature_depth(feature_depth), required_output(output_vector_size, 1){
    this->input = new Matrix* [feature_depth];
    for(int i = 0; i < feature_depth; i++){
        input[i] = new Matrix(input_vector_row, input_vector_col);
    }
}

Data_Loader::~Data_Loader(){
    for(int i = 0; i < this->feature_depth; i++){
        delete this->input[i];
    }
    delete[] this->input;
}

void Data_Loader::load_MNIST(std::ifstream &input, std::ifstream &required_output){
    for(int k = 0; k < this->feature_depth; k++){
        input.read((char*)this->input[k]->dv, this->input_vector_row * this->input_vector_col * sizeof(double));
    }
    required_output.read((char*)this->required_output.dv, this->output_vector_size  * sizeof(double));
}

void Data_Loader::load_CIFAR(std::ifstream &input){
    unsigned char tmp;
    input.read((char*)&tmp, sizeof(char));
    if(this->output_vector_size == 1){
        this->required_output.data[0][0] = tmp;
    }
    else
    {
        this->required_output.data[tmp][0] = 1;
    }
    int read_len = this->input_vector_row * this->input_vector_col;
    char temp_data[read_len];
    for(int k = 0; k < this->feature_depth; k++){
        input.read(temp_data, read_len);
        for(int i = 0; i < read_len; i++){
            this->input[k]->dv[i] = (double)temp_data[i]/255.0;
        }
    }
}

void Data_Loader::load_bmp(const char *path){
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
        if(signature == 19778){
            throw invalid_argument("The file is not a BMP file!\n");
        }
        char temp_inp[4 * this->input_vector_row * this->input_vector_col];
        inp.seekg(data_offset);
        unsigned char tmp;
        inp.read(temp_inp, this->input_vector_row * this->input_vector_col * 4);
        for(int i = 0; i < this->input_vector_row; i++){
            for(int j = 0; j < this->input_vector_col; j++){
                tmp = temp_inp[4*(i * this->input_vector_col + j)];
                this->input[0]->data[i][j] = tmp;
                tmp = temp_inp[4*(i * this->input_vector_col + j) + 1];
                this->input[1]->data[i][j] = tmp;
                tmp = temp_inp[4*(i * this->input_vector_col + j) + 2];
                this->input[2]->data[i][j] = tmp;
            }
        }
    }
    catch( exception &e){
        inp.close();
        throw e;
    }
    if(inp.is_open()){
        inp.close();
    }
}

void Data_Loader::load_sdl_pixels(SDL_Surface *window_surface)
{
    /*char temp_inp[4 * this->input_vector_row * this->input_vector_col];
    in.seekg(0x7a);
    in.read(temp_inp, this->input_vector_row * this->input_vector_col * 4);*/
    unsigned char* pixels = (unsigned char*)window_surface -> pixels;
    unsigned char tmp;
    for(int i = 0; i < this->input_vector_row; i++)
    {
        for(int j = 0; j < this->input_vector_col; j++)
        {
            tmp = pixels[4 * (i * window_surface -> w + j)];
            this->input[0]->data[i][j] = tmp;
            tmp = pixels[4 * (i * window_surface -> w + j) + 1];
            this->input[1]->data[i][j] = tmp;
            tmp = pixels[4 * (i * window_surface -> w + j) + 2];
            this->input[2]->data[i][j] = tmp;
            //tmp = pixels[4 * (i * window_surface -> w + j) + 3];
            //this->input[3]->data[i][j] = tmp;
        }
    }
}

