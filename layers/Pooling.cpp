#include "layers.h"
#include "Convolutional.h"
#include "Pooling.h"
#include <string.h>

Pooling::Pooling(Layer **network_layers, vector<int> input_from, int row, int col, int pooling_type, int my_index):
                map_row(row), map_col(col), pooling_type(pooling_type), my_index(my_index){
    if(input_from.size() > 1){
        throw runtime_error("Pooling layer must have only 1 input\n");
    }
    this->gets_input_from_ = input_from;
    this->network_layers = network_layers;
    this->input_row = this->network_layers[input_from[0]]->get_output_row();
    this->input_col = this->network_layers[input_from[0]]->get_output_col();
    this->fmap_count = this->network_layers[input_from[0]]->get_mapcount();
    this->output_row = input_row / row;
    if(input_row % row)
        output_row++;
    this->output_col = input_col / col;
    if(input_col % col)
        output_col++;
    this->threadcount = 1;
    this->build_outputs_and_errors();
}

Pooling::~Pooling(){
    this->destory_outputs_and_erros();
}

void Pooling::destory_outputs_and_erros(){
    for(int i = 0; i < this->threadcount; i++){
        for(int j = 0; j < this->fmap_count; j++){
            delete this->outputs[i][j];
            delete this->pooling_memory[i][j];
            delete this->layers_delta[i][j];
            delete this->layers_delta_helper[i][j];
        }
        delete[] this->outputs[i];
        delete[] this->flattened_output[i];
        delete[] this->pooling_memory[i];
        delete[] this->layers_delta[i];
        delete[] this->layers_delta_helper[i];
    }
    delete[] this->outputs;
    delete[] this->flattened_output;
    delete[] this->pooling_memory;
    delete[] this->layers_delta;
    delete[] this->layers_delta_helper;

    delete this->backprop_helper;
}

void Pooling::build_outputs_and_errors(){
    this->backprop_helper = new conv_backprop_helper(this->threadcount, this->output_row, this->output_col);

    this->outputs = new Matrix** [this->threadcount];
    this->flattened_output = new Matrix** [this->threadcount];
    this->pooling_memory = new Matrix** [this->threadcount];
    this->layers_delta = new Matrix** [this->threadcount];
    this->layers_delta_helper = new Matrix** [this->threadcount];
    for(int i = 0; i < this->threadcount; i++){
        this->outputs[i] = new Matrix* [this->fmap_count];
        this->flattened_output[i] = new Matrix* [1];
        this->pooling_memory[i] = new Matrix* [this->fmap_count];
        this->layers_delta[i] = new Matrix* [this->fmap_count];
        this->layers_delta_helper[i] = new Matrix* [this->fmap_count];
        this->flattened_output[i][0] = new Matrix(this->fmap_count * this->output_row * this->output_col, 1);
        for(int j = 0; j < this->fmap_count; j++){
            this->pooling_memory[i][j] = new Matrix(input_row, input_col);
            this->layers_delta[i][j] = new Matrix(input_row, input_col);
            this->layers_delta_helper[i][j] = new Matrix(this->output_row, this->output_col);
            this->outputs[i][j] = new Matrix(this->output_row, this->output_col);
        }
    }
}

inline void Pooling::max_pooling(Matrix **inpput, int threadindex){
    int in_row = this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[0][0].get_row();
    int in_col = this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[0][0].get_col();
    int input_r_index, input_c_index, max_r_index, max_c_index;
    double max;
    input_c_index = input_r_index = 0;
    for(int mapindex = 0; mapindex < this->fmap_count; mapindex++){
        this->pooling_memory[threadindex][mapindex][0].zero();
        for(int output_r = 0; output_r < this->output_row; output_r++){
            for(int output_c = 0; output_c < this->output_col; output_c++){
                max_r_index = input_r_index;
                max_c_index = input_c_index;
                max = this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[mapindex]->data[max_r_index][max_c_index];
                for(int map_r_index = 0; map_r_index < this->map_row and (input_r_index + map_r_index) < in_row; map_r_index++){
                    for(int map_c_index = 0; map_c_index < this->map_col and (input_c_index + map_c_index) < in_col; map_c_index++){
                        if(max < this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[mapindex]->data[input_r_index + map_r_index][input_c_index + map_c_index]){
                            max = this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[mapindex]->data[input_r_index + map_r_index][input_c_index + map_c_index];
                            max_r_index = input_r_index + map_r_index;
                            max_c_index = input_c_index + map_c_index;
                        }
                    }
                }
                this->outputs[threadindex][mapindex]->data[output_r][output_c] = max;
                this->pooling_memory[threadindex][mapindex]->data[max_r_index][max_c_index] = 1;
                input_c_index += this->map_col;
            }
            input_c_index = 0;
            input_r_index += this->map_row;
        }
        input_r_index = 0;
    }
}

inline Matrix** Pooling::backpropagate(Matrix **inpput, Layer *next_layer, Feature_map **nabla, Matrix ***delta, int threadindex){
    int in_row = this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[0][0].get_row();
    int in_col = this->network_layers[this->gets_input_from_[0]]->get_output(threadindex)[0][0].get_col();
    int delta_r_index, delta_c_index;
    delta_c_index = delta_r_index = 0;
    Feature_map** next_layers_fmaps;
    int next_layers_fmapcount, delta_index;
    this->backprop_helper->set_padded_delta_2d(delta, this->sends_output_to_, this->network_layers, threadindex);
    for(int i = 0; i < this->fmap_count; i++){
        this->layers_delta_helper[threadindex][i][0].zero();
        delta_index = 0;
        for(int next_layer_index : this->sends_output_to_){
            next_layers_fmaps = this->network_layers[next_layer_index]->get_feature_maps();
            next_layers_fmapcount = this->network_layers[next_layer_index]->get_mapcount();
            for(int j = 0; j < next_layers_fmapcount; j++){
                full_depth_cross_correlation(this->backprop_helper->padded_delta[threadindex][delta_index][j][0],
                                            next_layers_fmaps[j]->weights[0][0],
                                            this->layers_delta_helper[threadindex][i][0],
                                            1, 1);
            }
            delta_index++;
        }
    }
    this->backprop_helper->zero(threadindex);

    for(int mapindex = 0; mapindex < this->fmap_count; mapindex++){
        this->layers_delta[threadindex][mapindex][0].zero();
        for(int output_r = 0; output_r < this->output_row; output_r++){
            for(int output_c = 0; output_c < this->output_col; output_c++){
                for(int map_r_index = 0; map_r_index < this->map_row and (delta_r_index + map_r_index) < in_row; map_r_index++){
                    for(int map_c_index = 0; map_c_index < this->map_col and (delta_c_index + map_c_index) < in_col; map_c_index++){
                        if(pooling_memory[threadindex][mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] == 1){
                            layers_delta[threadindex][mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] = this->layers_delta_helper[threadindex][mapindex]->data[output_r][output_c];
                        }
                        else
                        {
                            layers_delta[threadindex][mapindex]->data[delta_r_index + map_r_index][delta_c_index + map_c_index] = 0;
                        }
                    }
                }
                delta_c_index += this->map_col;
            }
            delta_c_index = 0;
            delta_r_index += this->map_row;
        }
    }
    delta[this->my_index] = layers_delta[threadindex];
    return NULL;
}

void Pooling::layers_output(Matrix **input, int threadindex){
    switch(this->pooling_type){
    case MAX_POOLING:
        this->max_pooling(input, threadindex);
        break;
    default:
        cerr << "Unknown pooling type" << endl;
        throw exception();
    }
}

void Pooling::set_threadcount(int threadcnt){

    this->destory_outputs_and_erros();
    this->threadcount = threadcnt;
    this->build_outputs_and_errors();

}

int Pooling::get_threadcount(){
    return this->threadcount;
}

Matrix* Pooling::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex){
    cerr << "Pooling layer cannot be the last layer!" << endl;
    throw exception();
}

Matrix** Pooling::derivate_layers_output(Matrix **input, int threadindex){
    cerr << "The output of pooling layer cannot be derived!" << endl;
    throw exception();
}

void Pooling::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer){
    return;
}

inline void Pooling::remove_some_neurons(Matrix ***w_bckup, Matrix ***b_bckup, int **layers_bckup, int ***indexes){
    return;
}

inline void Pooling::add_back_removed_neurons(Matrix **w_bckup, Matrix **b_bckup, int *layers_bckup, int **indexes){
    return;
}

void Pooling::set_input(Matrix **input, int threadindex){
    cerr << "Pooling layer cannot be an input layer! Set input is not possible." << endl;
    throw exception();
}

Matrix** Pooling::get_output(int threadindex){
    return this->outputs[threadindex];
}

void Pooling::create_connections(vector<int> input_from, vector<int> output_to){
    ///TODO some error checking
    this->gets_input_from_ = input_from;
    this->sends_output_to_ = output_to;
}

const vector<int>& Pooling::gets_input_from() const
{
    return this->gets_input_from_;
}

const vector<int>& Pooling::sends_output_to() const
{
    return this->sends_output_to_;
}

inline Feature_map** Pooling::get_feature_maps(){
    cerr << "Pooling layer doesn't have feature maps" << endl;
    throw exception();
}

inline short Pooling::get_layer_type(){
    return POOLING;
}

inline int Pooling::get_output_len(){
    return (this->output_row * this->output_col * this->fmap_count);
}

inline int Pooling::get_output_row(){
    return this->output_row;
}

inline int Pooling::get_output_col(){
    return this->output_col;
}

int Pooling::get_mapcount(){
    return this->fmap_count;
}

int Pooling::get_mapdepth(){
    return 1;
}

int Pooling::get_weights_row(){
    return this->map_row;
}

int Pooling::get_weights_col(){
    return this->map_col;
}

void Pooling::store(std::ofstream &params){
    ;
}

void Pooling::load(std::ifstream &params){
    ;
}

