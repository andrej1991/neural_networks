#ifndef DATA_LOADER_H_INCLUDED
#define DATA_LOADER_H_INCLUDED

#include <fstream>
#include "../matrix/matrix.h"
#include <SDL2/SDL.h>

class Data_Loader{
    public:
    Matrix **input;
    Matrix required_output;
    int input_vector_row, input_vector_col, feature_depth, output_vector_size;
    Data_Loader(int input_row, int input_col, int output_vector_size, int feature_depth);
    ~Data_Loader();
    void load_MNIST(std::ifstream &input, std::ifstream &required_output);
    void load_CIFAR(std::ifstream &input);
    void load_bmp(const char *path);
    void load_sdl_pixels(SDL_Surface *window_surface);

};


#endif // DATA_LOADER_H_INCLUDED
