#include <string.h>
#include "layers.h"

Flatten::Flatten(Layer **layers, int index, vector<int> input_from, int mapcount_) : my_index(index), map_count(mapcount_)
{
    this->gets_input_from_ = input_from;
    this->network_layers = layers;
    this->threadcount = 1;
    this->outputlen = 0;
    this->fmap = NULL;
    for(int i : input_from)
    {
        this->outputlen += this->network_layers[i]->get_output_len();
    }
    this->build_dinamic_data();
}

Flatten::~Flatten()
{
    this->destroy_dinamic_data();
}

void Flatten::destroy_dinamic_data()
{
    for(int i = 0; i < this->threadcount; i++)
    {
        delete this->output[i][0];
        delete[] this->output[i];
    }
    delete[] this->output;
}
void Flatten::build_dinamic_data()
{
    this->output = new Matrix**[this->threadcount];
    this->layers_delta = new Matrix**[this->threadcount];
    for(int i = 0; i < this->threadcount; i++)
    {
        this->output[i] = new Matrix* [1];
        this->layers_delta[i] = new Matrix* [this->map_count];
        this->output[i][0] = new Matrix(this->outputlen, 1);
        for(int j = 0; j < this->map_count; j++)
            this->layers_delta[i][j] = new Matrix(1, 1);
    }
}

Matrix** Flatten::backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix ***next_layers_error, int threadindex)
{
    for(int i = 0; i < this->map_count; i++)
    {
        this->layers_delta[threadindex][i]->data[0][0] = next_layers_error[this->sends_output_to_[0]][0][0].data[i][0];
    }
    next_layers_error[this->my_index] = this->layers_delta[threadindex];
    return NULL;
}

void Flatten::layers_output(Matrix **input, int threadindex)
{
    for(int i : this->gets_input_from_)
    {
        int map_count_ = this->network_layers[i]->get_mapcount();
        int output_size = this->network_layers[i]->get_output_row() * this->network_layers[i]->get_output_col();
        int output_size_in_bytes = output_size * sizeof(double);
        for(int map_index = 0; map_index < map_count_; map_index++)
        {
            memcpy(&(this->output[threadindex][0]->dv[map_index*output_size]), this->network_layers[i]->get_output(threadindex)[map_index]->dv, output_size_in_bytes);
        }
    }
}

Matrix* Flatten::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    throw runtime_error("currently the Flatten layer needs to be connected to a fully connected or a softmax layer\n");
}

Matrix** Flatten::derivate_layers_output(Matrix **input, int threadindex)
{
    ;
}

void Flatten::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    ;
}

void Flatten::set_input(Matrix **input, int threadindex)
{
    throw runtime_error("Flatten layer cannot be an input layer\n");
}

inline Matrix** Flatten::get_output(int threadindex)
{
    return this->output[threadindex];
}

inline Feature_map** Flatten::get_feature_maps()
{
    if(this->fmap == NULL)
    {
        this->map_count = this->network_layers[this->sends_output_to_[0]]->get_output_len();
        this->fmap = new Feature_map* [this->map_count];
        for(int i = 0; i < this->map_count; i++)
        {
            this->fmap[i] = new Feature_map(this->get_weights_row(), this->get_weights_col(), this->network_layers[this->gets_input_from_[0]]->get_mapcount());
        }
    }
    int input_from_me = 0;
    for(int k : this->network_layers[this->sends_output_to_[0]]->gets_input_from())
    {
        if(k == this->my_index)
        {
            break;
        }
        input_from_me++;
    }
    Feature_map *next_layers_fmap = this->network_layers[this->sends_output_to_[0]]->get_feature_maps()[input_from_me];
    int kernelsize = this->get_weights_row() * this->get_weights_col();
    for(int i = 0; i < this->map_count; i++)
    {
        for(int j = 0; j < this->fmap[0]->get_mapdepth(); j++)
        {
            int starting_pos = kernelsize * j;
            memcpy(this->fmap[i]->weights[j]->dv, &(next_layers_fmap->weights[0]->data[i][starting_pos]), kernelsize*sizeof(double));
        }
    }
    return this->fmap;
}

inline short Flatten::get_layer_type()
{
    return FLATTEN;
}

inline int Flatten::get_output_len()
{
    return this->outputlen;
}

inline int Flatten::get_output_row()
{
    return this->outputlen;
}

inline int Flatten::get_output_col()
{
    return 1;
}

int Flatten::get_mapcount()
{
    return this->map_count;
}

int Flatten::get_mapdepth()
{
    return this->fmap[0]->get_mapdepth();
}

int Flatten::get_weights_row()
{
    return this->network_layers[this->gets_input_from_[0]]->get_output_row();
}

int Flatten::get_weights_col()
{
    return this->network_layers[this->gets_input_from_[0]]->get_output_col();
}

void Flatten::drop_out_some_neurons(double probability, Matrix **colums_to_remove)
{
    this->backup_map_count = this->map_count;
    this->map_count = this->network_layers[this->sends_output_to_[0]]->get_mapcount();
    this->backup_layers_delta = this->layers_delta;
    this->layers_delta = new Matrix**[this->threadcount];
    for(int i = 0; i < this->threadcount; i++)
    {
        this->layers_delta[i] = new Matrix* [this->map_count];
        for(int j = 0; j < this->map_count; j++)
            this->layers_delta[i][j] = new Matrix(1, 1);
    }
    return;
}

void Flatten::restore_neurons(Matrix **removed_colums)
{
    for(int i = 0; i < this->threadcount; i++)
    {
        for(int j = 0; j < this->map_count; j++)
            delete this->layers_delta[i][j];
        delete[] this->layers_delta[i];
    }
    delete[] this->layers_delta;
    this->map_count = this->backup_map_count;
    this->layers_delta = this->backup_layers_delta;
}

void Flatten::store(std::ofstream &params)
{
    ;
}

void Flatten::load(std::ifstream &params)
{
    ;
}

void Flatten::set_threadcount(int threadcount_)
{
    this->destroy_dinamic_data();
    this->threadcount = threadcount_;
    this->build_dinamic_data();
}

inline int Flatten::get_threadcount()
{
    return this->threadcount;
}

void Flatten::create_connections(vector<int> input_from, vector<int> output_to)
{
    this->gets_input_from_ = input_from;
    this->sends_output_to_ = output_to;
}

const vector<int>& Flatten::gets_input_from() const
{
    return this->gets_input_from_;
}

const vector<int>& Flatten::sends_output_to() const
{
    return this->sends_output_to_;
}

int Flatten::get_vertical_stride()
{
    return 1;
}

int Flatten::get_horizontal_stride()
{
    return 1;
}
