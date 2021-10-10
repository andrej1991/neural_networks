#include "layers.h"
#include "Convolutional.h"
#include <string.h>

Pooling::Pooling(int row, int col, int pooling_type, int prev_layers_fmapcount, int input_row, int input_col, int next_layers_type):
                fmap_count(prev_layers_fmapcount), map_row(row), map_col(col), pooling_type(pooling_type), next_layers_type(next_layers_type)
{
    this->output_row = input_row / row;
    if(input_row % row)
        output_row++;
    this->output_col = input_col / col;
    if(input_col % col)
        output_col++;
    this->pooling_memory = new Matrix* [prev_layers_fmapcount];
    this->layers_delta = new Matrix* [prev_layers_fmapcount];
    this->layers_delta_helper = new Matrix* [prev_layers_fmapcount];
    this->outputs = new Matrix* [prev_layers_fmapcount];
    this->flattened_output = new Matrix* [1];
    this->flattened_output[0] = new Matrix(this->fmap_count * this->output_row * this->output_col, 1);
    for(int i = 0; i < prev_layers_fmapcount; i++)
    {
        this->pooling_memory[i] = new Matrix(input_row, input_col);
        this->layers_delta[i] = new Matrix(input_row, input_col);
        this->layers_delta_helper[i] = new Matrix(this->output_row, this->output_col);
        this->outputs[i] = new Matrix(this->output_row, this->output_col);
    }
}

Pooling::~Pooling()
{
    for(int i = 0; i < this->fmap_count; i++)
    {
        delete this->pooling_memory[i];
        delete this->layers_delta[i];
        delete this->outputs[i];
    }
    delete[] this->pooling_memory;
    delete[] this->layers_delta;
    delete[] this->outputs;
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
                this->outputs[mapindex]->data[output_r][output_c] = max;
                this->pooling_memory[mapindex]->data[max_r_index][max_c_index] = 1;
                input_c_index += this->map_col;
            }
            input_c_index = 0;
            input_r_index += this->map_row;
        }
        input_r_index = 0;
    }
}

inline void delete_padded_delta(Matrix **padded_delta, int limit)
{
    for(int i = 0; i < limit; i++)
    {
        delete padded_delta[i];
    }
    delete[] padded_delta;
}

void Pooling::get_2D_weights(int neuron_id, int fmap_id, Matrix &kernel, Feature_map **next_layers_fmap)
{
    int kernelsize = kernel.get_row() * kernel.get_col();
    int starting_pos = kernelsize * fmap_id;
    int index = starting_pos;
    memcpy(kernel.dv, &(next_layers_fmap[0]->weights[0]->data[neuron_id][starting_pos]), kernelsize*sizeof(double));
}

inline Matrix** Pooling::backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix **delta, int threadindex)
{
    int in_row = input[0][0].get_row();
    int in_col = input[0][0].get_col();
    int delta_r_index, delta_c_index;
    double max;
    delta_c_index = delta_r_index = 0;
    Feature_map** next_layers_fmaps = next_layer->get_feature_maps();
    int next_layers_fmapcount = next_layer->get_mapcount();
    Matrix **padded_delta;
    Matrix helper(this->output_row, this->output_col);
    Matrix dilated;
    if(next_layers_type == FULLY_CONNECTED or next_layers_type == SOFTMAX)
    {
        int next_layers_neuroncount = delta[0]->get_row();
        padded_delta = new Matrix* [next_layers_neuroncount];
        for(int i = 0; i < next_layers_neuroncount; i++)
        {
            padded_delta[i] = new Matrix;
            padded_delta[i][0].data[0][0] = delta[0][0].data[i][0];
            padded_delta[i][0] = padded_delta[i][0].zero_padd((this->output_row-1)/2,
                                                     (this->output_col-1)/2,
                                                     (this->output_row-1)/2,
                                                     (this->output_col-1)/2);
        }
        Matrix kernel(this->output_row, this->output_col);
        for(int i = 0; i < this->fmap_count; i++)
        {
            this->layers_delta_helper[i][0].zero();
            for(int j = 0; j < next_layers_neuroncount; j++)
            {
                this->get_2D_weights(j, i, kernel, next_layers_fmaps);
                //convolution(padded_delta[0],kernel, helper);
                cross_correlation(padded_delta[j][0],kernel, helper, 1, 1);
                this->layers_delta_helper[i][0] += helper;
            }
        }
        delete_padded_delta(padded_delta, next_layers_neuroncount);
    }
    else if(next_layers_type == CONVOLUTIONAL)
    {
        padded_delta = new Matrix* [next_layers_fmapcount];
        for(int i = 0; i < next_layers_fmapcount; i++)
        {
            padded_delta[i] = new Matrix;
            dilated = delta[i][0].dilate(static_cast<Convolutional*>(next_layer)->get_vertical_stride(), static_cast<Convolutional*>(next_layer)->get_horizontal_stride());
            padded_delta[i][0] = dilated.zero_padd((next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2);
        }
        for(int i = 0; i < this->fmap_count; i++)
        {
            this->layers_delta_helper[i][0].zero();
            for(int j = 0; j < next_layers_fmapcount; j++)
            {
                cross_correlation(padded_delta[j][0], next_layers_fmaps[j]->weights[i][0], helper, 1, 1);
                this->layers_delta_helper[i][0] += helper;
            }
        }
        delete_padded_delta(padded_delta, next_layers_fmapcount);
    }
    else if(next_layers_type == POOLING)
    {
        cerr << "Two poolin layer cannot follow each other!\n";
        throw exception();
    }
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
                            layers_delta[mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] = this->layers_delta_helper[mapindex]->data[output_r][output_c];
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

void Pooling::layers_output(Matrix **input, int threadindex)
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

Matrix** Pooling::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    cerr << "Pooling layer cannot be the last layer!" << endl;
    throw exception();
}

Matrix** Pooling::derivate_layers_output(Matrix **input, int threadindex)
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

void Pooling::set_input(Matrix **input, int threadindex)
{
    cerr << "Pooling layer cannot be an input layer! Set input is not possible." << endl;
    throw exception();
}

void Pooling::flatten()
{
    int i = 0;
    int output_size = this->output_row * this->output_col;
    int output_size_in_bytes = output_size * sizeof(double);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        memcpy(&(this->flattened_output[0]->dv[map_index*output_size]), this->outputs[map_index]->dv, output_size_in_bytes);
    }
}

Matrix** Pooling::get_output(int threadindex)
{
    if(this->next_layers_type == FULLY_CONNECTED)
    {
        this->flatten();
        return this->flattened_output;
    }
    else
        return this->outputs;
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
