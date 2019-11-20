#include "layers.h"

Pooling::Pooling(int row, int col, int pooling_type, int prev_layers_fmapcount, int input_row, int input_col): fmap_count(prev_layers_fmapcount), map_row(row), map_col(col), pooling_type(pooling_type)
{
    this->output_row = input_row / row;
    if(input_row % row)
        output_row++;
    this->output_col = input_col / col;
    if(input_col % col)
        output_col++;
    this->pooling_memory = new Matrix* [prev_layers_fmapcount];
    this->layers_delta = new Matrix* [prev_layers_fmapcount];
    this->output = new Matrix* [prev_layers_fmapcount];
    for(int i = 0; i < prev_layers_fmapcount; i++)
    {
        this->pooling_memory[i] = new Matrix(input_row, input_col);
        this->layers_delta[i] = new Matrix(input_row, input_col);
        this->output[i] = new Matrix(this->output_row, this->output_col);
    }
}

Pooling::~Pooling()
{
    for(int i = 0; i < this->fmap_count; i++)
    {
        delete this->pooling_memory[i];
        delete this->layers_delta[i];
        delete this->output[i];
    }
    delete[] this->pooling_memory;
    delete[] this->layers_delta;
    delete[] this->output;
}

inline void Pooling::max_pooling(Matrix **input)
{
    int in_row = input[0][0].get_row();
    int in_col = input[0][0].get_col();
    int input_r_index, input_c_index, max_r_index, max_c_index;
    double max;
    input_c_index = input_r_index = 0;
    for(int mapindex = 0; mapindex < this->fmap_count; mapindex++)
    {
        this->pooling_memory[mapindex][0].zero();
        for(int output_r = 0; output_r < this->output_row; output_r++)
        {
            for(int output_c = 0; output_c < this->output_col; output_c++)
            {
                max_r_index = input_r_index;
                max_c_index = input_c_index;
                max = input[mapindex]->data[max_r_index][max_c_index];
                for(int map_r_index = 0; map_r_index < this->map_row and (input_r_index + map_r_index) < in_row; map_r_index++)
                {
                    for(int map_c_index = 0; map_c_index < this->map_col and (input_c_index + map_c_index) < in_col; map_c_index++)
                    {
                        if(max < input[mapindex]->data[input_r_index + map_r_index][input_c_index + map_c_index])
                        {
                            max = input[mapindex]->data[input_r_index + map_r_index][input_c_index + map_c_index];
                            max_r_index = input_r_index + map_r_index;
                            max_c_index = input_c_index + map_c_index;
                        }
                    }
                }
                this->output[mapindex]->data[output_r][output_c] = max;
                this->pooling_memory[mapindex]->data[max_r_index][max_c_index] = 1;
                input_c_index += this->map_col;
            }
            input_c_index = 0;
            input_r_index += this->map_row;
        }
    }
}

inline Matrix** Pooling::backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix **next_layers_error)
{
    int in_row = input[0][0].get_row();
    int in_col = input[0][0].get_col();
    int delta_r_index, delta_c_index;
    double max;
    delta_c_index = delta_r_index = 0;
    for(int mapindex = 0; mapindex < this->fmap_count; mapindex++)
    {
        this->layers_delta[mapindex][0].zero();
        for(int output_r = 0; output_r < this->output_row; output_r++)
        {
            for(int output_c = 0; output_c < this->output_col; output_c++)
            {
                for(int map_r_index = 0; map_r_index < this->map_row and (delta_r_index + map_r_index) < in_row; map_r_index++)
                {
                    for(int map_c_index = 0; map_c_index < this->map_col and (delta_c_index + map_c_index) < in_col; map_c_index++)
                    {
                        if(pooling_memory[mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] == 1)
                        {
                            layers_delta[mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] = next_layers_error[mapindex]->data[output_r][output_c];
                        }
                        else
                        {
                            layers_delta[mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] = 0;
                        }
                    }
                }
                delta_c_index += this->map_col;
            }
            delta_c_index = 0;
            delta_r_index += this->map_row;
        }
    }
    return this->layers_delta;
}

inline void Pooling::layers_output(Matrix **input)
{
    switch(this->pooling_type)
    {
    case MAX_POOLING:
        this->max_pooling(input);
        break;
    default:
        cerr << "Unknown pooling type" << endl;
        throw exception();
    }
}

inline Matrix** Pooling::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type)
{
    cerr << "Pooling layer cannot be the last layer!" << endl;
    throw exception();
}

inline Matrix** Pooling::derivate_layers_output(Matrix **input)
{
    cerr << "The output of pooling layer cannot be derived!" << endl;
    throw exception();
}

void Pooling::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    return;
}

inline void Pooling::remove_some_neurons(Matrix ***w_bckup, Matrix ***b_bckup, int **layers_bckup, int ***indexes)
{
    return;
}

inline void Pooling::add_back_removed_neurons(Matrix **w_bckup, Matrix **b_bckup, int *layers_bckup, int **indexes)
{
    return;
}

void Pooling::set_input(Matrix **input)
{
    cerr << "Pooling layer cannot be an inpu layer! Set input is not possible." << endl;
    throw exception();
}

inline Matrix** Pooling::get_output()
{
    return this->output;
}

inline Feature_map** Pooling::get_feature_maps()
{
    cerr << "Pooling layer doesn't have feature maps" << endl;
    throw exception();
}

inline short Pooling::get_layer_type()
{
    return POOLING;
}

inline int Pooling::get_output_len()
{
    return (this->output_row * this->output_col * this->fmap_count);
}

inline int Pooling::get_output_row()
{
    return this->output_row;
}

inline int Pooling::get_output_col()
{
    return this->output_col;
}

int Pooling::get_mapcount()
{
    return this->fmap_count;
}

int Pooling::get_mapdepth()
{
    return 1;
}

int Pooling::get_weights_row()
{
    return this->map_row;
}

int Pooling::get_weights_col()
{
    return this->map_col;
}

void Pooling::store(std::ofstream &params)
{
    ;
}

void Pooling::load(std::ifstream &params)
{
    ;
}
