#ifndef CONVOLUTIONAL_H_INCLUDED
#define CONVOLUTIONAL_H_INCLUDED

#include "layers.h"
#include <vector>

class conv_output_helper{
    int threadcount;
    public:
    Matrix **convolved, **helper;
    conv_output_helper(int threadcnt, int row, int col);
    ~conv_output_helper();
};

class conv_backprop_helper{
    int threadcount, outputcount;
    int **layer_count;
    public:
    Matrix ****padded_delta, **helper, *dilated, **kernel;
    conv_backprop_helper(int threadcnt, int row, int col);
    ~conv_backprop_helper();
    void set_padded_delta_1d(Matrix **delta, int next_layers_neuroncount, int top, int right, int bottom, int left, int threadcnt);
    void set_padded_delta_2d(Matrix ***delta, std::vector<int> sends_output, Layer **network_layers, int threadcnt);
    void delete_padded_delta(int threadindx);
    void zero(int threadid);
    int get_layercount(int threadidx);
};

class Convolutional : public Layer {
    Matrix ***outputs, ***flattened_output, ***layers_delta, ***output_derivative, ***layers_delta_helper;
    Feature_map **fmap;
    Padding pad;
    conv_output_helper *feedforward_helpter;
    conv_backprop_helper *backprop_helper;
    int neuron_type, outputlen, input_row, input_col, kernel_row, kernel_col, map_count, vertical_stride, horizontal_stride, next_layers_type, output_row, output_col, threadcount, input_channel_count, my_index;
    short int layer_type;
    Neuron neuron;
    void fulldepth_conv(Matrix &helper, Matrix &convolved, Matrix **input, int map_index);
    void destory_outputs_and_erros();
    void build_outputs_and_errors();
    vector<int> gets_input_from_, sends_output_to_;
    Layer **network_layers;
    public:
    Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, int my_index_, Padding &p, int vertical_stride = 1, int horizontal_stride = 1);
    ~Convolutional();
    Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***next_layers_error, int threadindex);
    void layers_output(Matrix **input, int threadindex);
    Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex);
    Matrix** derivate_layers_output(Matrix **input, int threadindex);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    void set_input(Matrix **input, int threadindex);
    Matrix** get_output(int threadindex = 0);
    inline Feature_map** get_feature_maps();
    inline short get_layer_type();
    inline int get_output_len();
    inline int get_output_row();
    inline int get_output_col();
    void flatten(int threadindex);
    int get_mapcount();
    int get_mapdepth();
    int get_weights_row();
    int get_weights_col();
    int get_vertical_stride();
    int get_horizontal_stride();
    void get_2D_weights(int neuron_id, int fmap_id, Matrix &kernel, Feature_map **next_layers_fmap);
    Matrix drop_out_some_neurons(double probability = 0, Matrix *colums_to_remove = NULL){};
    void restore_neurons(Matrix *removed_colums = NULL){};
    void store(std::ofstream &params);
    void load(std::ifstream &params);
    virtual void set_threadcount(int threadcnt, vector<Matrix***> inputs_);
    virtual int get_threadcount();
    virtual void create_connections(vector<int> input_from, vector<int> output_to);
    virtual const vector<int>& gets_input_from() const;
    virtual const vector<int>& sends_output_to() const;
    virtual void set_graph_information(Layer **network_layers, int my_index);
};

#endif // CONVOLUTIONAL_H_INCLUDED
