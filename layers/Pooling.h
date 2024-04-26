#ifndef POOLING_H_INCLUDED
#define POOLING_H_INCLUDED

#include "Convolutional.h"

class Pooling : public Layer {
    Matrix ***outputs, ***pooling_memory, ***layers_delta, ***layers_delta_helper, ***flattened_output;
    int fmap_count, output_row, output_col, map_row, map_col, pooling_type, next_layers_type, threadcount;
    int input_row, input_col;
    conv_backprop_helper *backprop_helper;
    inline void max_pooling(Matrix **input, int threadindex);
    void get_2D_weights(int neuron_id, int fmap_id, Matrix &kernel, Feature_map **next_layers_fmap);
    void flatten(int threadindex);
    void destory_outputs_and_erros();
    void build_outputs_and_errors();
    vector<int> gets_input_from_, sends_output_to_;
    public:
    Pooling(int row, int col, int pooling_type, int prev_layers_fmapcount, int input_row , int input_col, int next_layers_type);
    ~Pooling();
    Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix ***next_layers_error, int threadindex);
    void layers_output(Matrix **input, int threadindex);
    Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex);
    Matrix** derivate_layers_output(Matrix **input, int threadindex);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    inline void remove_some_neurons(Matrix ***w_bckup, Matrix ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrix **w_bckup, Matrix **b_bckup, int *layers_bckup, int **indexes);
    void set_input(Matrix **input, int threadindex);
    Matrix** get_output(int threadindex = 0);
    inline Feature_map** get_feature_maps();
    inline short get_layer_type();
    inline int get_output_len();
    inline int get_output_row();
    inline int get_output_col();
    int get_mapcount();
    int get_mapdepth();
    int get_weights_row();
    int get_weights_col();
    Matrix drop_out_some_neurons(double probability = 0, Matrix *colums_to_remove = NULL){};
    void restore_neurons(Matrix *removed_colums = NULL){};
    void store(std::ofstream &params);
    void load(std::ifstream &params);
    virtual void set_threadcount(int threadcnt, vector<Matrix***> inputs_);
    virtual int get_threadcount();
    virtual void create_connections(vector<int> input_from, vector<int> output_to);
    virtual const vector<int>& gets_input_from() const;
    virtual const vector<int>& sends_output_to() const;
};


#endif // POOLING_H_INCLUDED
