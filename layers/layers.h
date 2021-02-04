#ifndef LAYERS_H_INCLUDED
#define LAYERS_H_INCLUDED

#include <iostream>

#include "../neurons/neuron.h"
#include "../data_loader/MNIST_data.h"
#include "../matrix/matrix.h"

#define QUADRATIC_CF 0
#define CROSS_ENTROPY_CF 1
#define LOG_LIKELIHOOD_CF 2

#define INPUTLAYER -1
#define FULLY_CONNECTED 0
#define CONVOLUTIONAL 1
#define SOFTMAX 2
#define POOLING 3
#define MAX_POOLING 3

class LayerDescriptor{
    public:
    int layer_type;
    int neuron_count, row, col;
    int mapcount;
    int horizontal_stride, vertical_stride;
    int neuron_type;
    public:
    LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int vertical_stride = 1, int horizontal_stride = 1);
};

class Feature_map{
    int /*row, col,*/ mapdepth;
    public:
    Matrix **weights, **biases;
    Feature_map(int row, int col, int mapdepth, int biascnt = -1);
    int get_col();
    int get_row();
    int get_mapdepth();
    void initialize_biases(double standard_deviation = 1, double mean = 0);
    void initialize_weights(double standard_deviation = 1, double mean = 0);
    void store(std::ofstream &params);
    void load(std::ifstream &params);
    ~Feature_map();
};

class Layers_features{
    int fmap_count, biascnt;
    public:
    Feature_map **fmap;
    Layers_features(int mapcount, int row, int col, int depth, int biascnt);
    Layers_features(const Layers_features &layer);
    ~Layers_features();
    void operator+=(const Layers_features &layer);
    Layers_features operator+(const Layers_features &layer);
    Layers_features operator/(const Layers_features &layer);
    //Layers_features operator*(const Layers_features &layer);
    Layers_features & operator=(const Layers_features &layer);
    Layers_features operator*(double d);
    Layers_features operator+(double d);
    Layers_features sqroot();
    Layers_features square_element_by();
    void zero();
    int get_fmap_count();
};

class Padding{
    public:
    short int left_padding, right_padding, top_padding, bottom_padding;
    Padding(short int left = 0, short int top = 0, short int right = 0, short int bottom = 0) :
        left_padding(left), right_padding(right), top_padding(top), bottom_padding(bottom){}
};

class Layer{
    public:
    virtual ~Layer();
    virtual inline Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix **next_layers_error) = 0;
    virtual inline void layers_output(Matrix **input) = 0;
    virtual inline Matrix** get_output_error(Matrix **input, Matrix &required_output, int costfunction_type) = 0;
    virtual inline Matrix** derivate_layers_output(Matrix **input) = 0;
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer) = 0;
    virtual void set_input(Matrix **input) = 0;
    virtual inline Matrix** get_output() = 0;
    virtual inline Feature_map** get_feature_maps() = 0;
    virtual inline short get_layer_type() = 0;
    virtual inline int get_output_len() = 0;
    virtual inline int get_output_row() = 0;
    virtual inline int get_output_col() = 0;
    virtual int get_mapcount() = 0;
    virtual int get_mapdepth() = 0;
    virtual int get_weights_row() = 0;
    virtual int get_weights_col() = 0;
    virtual Matrix drop_out_some_neurons(double probability = 0, Matrix *colums_to_remove = NULL) = 0;
    virtual void restore_neurons(Matrix *removed_colums = NULL) = 0;
    virtual void store(std::ofstream &params) = 0;
    virtual void load(std::ifstream &params) = 0;

};

class FullyConnected : public Layer {
    friend class Softmax;
    Matrix **output, **output_derivative, **output_error, **output_error_helper, **layers_delta;
    Matrix *removed_rows, *backup_weights, *backup_biases, *backup_output, *backup_output_derivative, *backpup_output_error, *backup_output_error_helper, *backup_layers_delta;
    int neuron_type, outputlen, backup_outputlen;
    bool dropout_happened;
    short int layer_type;
    Neuron neuron;
    Feature_map **fmap;
    public:
    FullyConnected(int row, int prev_row, int neuron_type);
    ~FullyConnected();
    virtual inline Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error);
    virtual inline void layers_output(Matrix **input);
    virtual inline Matrix** get_output_error(Matrix **input, Matrix &required_output, int costfunction_type);
    virtual inline Matrix** derivate_layers_output(Matrix **input);
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    virtual void set_input(Matrix **input);
    virtual inline Matrix** get_output();
    virtual inline Feature_map** get_feature_maps();
    virtual inline short get_layer_type();
    virtual inline int get_output_len();
    virtual inline int get_output_row();
    virtual inline int get_output_col();
    virtual int get_mapcount();
    virtual int get_mapdepth();
    virtual int get_weights_row();
    virtual int get_weights_col();
    virtual Matrix drop_out_some_neurons(double probability = 0, Matrix *colums_to_remove = NULL);
    virtual void restore_neurons(Matrix *removed_colums = NULL);
    virtual void store(std::ofstream &params);
    virtual void load(std::ifstream &params);
};

class Softmax : public FullyConnected {
    public:
    Softmax(int row, int col);
    ~Softmax();
    inline Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error);
    inline void layers_output(Matrix **input);
    inline Matrix** get_output_error(Matrix **input, Matrix &required_output, int costfunction_type);
    inline Matrix** derivate_layers_output(Matrix **input);
};

class Convolutional : public Layer {
    Matrix **outputs, **flattened_output, **layers_delta, **output_derivative, **layers_delta_helper;
    Feature_map **fmap;
    Padding pad;
    int neuron_type, outputlen, input_row, input_col, kernel_row, kernel_col, map_count, vertical_stride, horizontal_stride, next_layers_type, output_row, output_col;
    short int layer_type;
    Neuron neuron;
    inline void fulldepth_conv(Matrix &helper, Matrix &convolved, Matrix **input, int map_index);
    public:
    Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int vertical_stride = 1, int horizontal_stride = 1);
    ~Convolutional();
    inline Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error);
    inline void layers_output(Matrix **input);
    inline Matrix** get_output_error(Matrix **input, Matrix &required_output, int costfunction_type);
    inline Matrix** derivate_layers_output(Matrix **input);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    void set_input(Matrix **input);
    inline Matrix** get_output();
    inline Feature_map** get_feature_maps();
    inline short get_layer_type();
    inline int get_output_len();
    inline int get_output_row();
    inline int get_output_col();
    void flatten();
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
};


class InputLayer : public Layer {
    public:
    short int layer_type, next_layers_type;
    int outputlen, row, col, input_channel_count;
    Matrix **outputs;
    Padding padd;
    InputLayer(int row, int col, int input_channel_count, int neuron_type, Padding &p, short int next_layers_type);
    ~InputLayer();
    inline Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error);
    inline void layers_output(Matrix **input);
    inline Matrix** get_output_error(Matrix **input, Matrix &required_output, int costfunction_type);
    inline Matrix** derivate_layers_output(Matrix **input);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    void set_input(Matrix **input);
    inline Matrix** get_output();
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
};

class Pooling : public Layer {
    Matrix **outputs, **pooling_memory, **layers_delta, **layers_delta_helper, **flattened_output;
    int fmap_count, output_row, output_col, map_row, map_col, pooling_type, next_layers_type;
    inline void max_pooling(Matrix **input);
    void get_2D_weights(int neuron_id, int fmap_id, Matrix &kernel, Feature_map **next_layers_fmap);
    void flatten();
    public:
    Pooling(int row, int col, int pooling_type, int prev_layers_fmapcount, int input_row , int input_col, int next_layers_type);
    ~Pooling();
    inline Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix **next_layers_error);
    inline void layers_output(Matrix **input);
    inline Matrix** get_output_error(Matrix **input, Matrix &required_output, int costfunction_type);
    inline Matrix** derivate_layers_output(Matrix **input);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    inline void remove_some_neurons(Matrix ***w_bckup, Matrix ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrix **w_bckup, Matrix **b_bckup, int *layers_bckup, int **indexes);
    void set_input(Matrix **input);
    inline Matrix** get_output();
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
};

#endif // LAYERS_H_INCLUDED
