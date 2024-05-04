#ifndef LAYERS_H_INCLUDED
#define LAYERS_H_INCLUDED

#include <iostream>
#include <string>
#include <vector>

#include "../neurons/neuron.h"
#include "../data_loader/data_loader.h"
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
#define FLATTEN 4

class LayerDescriptor{
    public:
    int layer_type;
    int neuron_count, row, col;
    int mapcount;
    int horizontal_stride, vertical_stride;
    int neuron_type;
    string name;
    std::vector<string> output_connections;
    LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int vertical_stride = 1, int horizontal_stride = 1);
    LayerDescriptor(int layer_type, int neuron_type, int neuron_count, std::vector<string> output_connections, string name = "", int col = 1, int mapcount = 1, int vertical_stride = 1, int horizontal_stride = 1);
    std::string get_name();
    std::vector<std::string> get_output_connections();
};

class Feature_map{
    int /*row, col,*/ mapdepth;
    public:
    Matrix **weights, **biases;
    Feature_map(int row, int col, int mapdepth, int biascnt = 1, int bias_col = 1);
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
    int fmap_count, biasrow, biascol;
    public:
    Feature_map **fmap;
    Layers_features(int mapcount, int row, int col, int depth, int biasrow, int biascol);
    Layers_features(int mapcount, Feature_map** fmap, int biasrow, int biascol);
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
    virtual Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix ***next_layers_error, int threadindex) = 0;
    virtual void layers_output(Matrix **input, int threadindex) = 0;
    virtual Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex) = 0;
    virtual Matrix** derivate_layers_output(Matrix **input, int threadindex) = 0;
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer) = 0;
    virtual void set_input(Matrix **input, int threadindex) = 0;
    virtual inline Matrix** get_output(int threadindex = 0) = 0;
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
    virtual void set_threadcount(int threadcount, vector<Matrix***> inputs_);
    virtual void set_threadcount(int threadcount);
    virtual inline int get_threadcount();
    virtual void create_connections(vector<int> input_from, vector<int> output_to);
    virtual const vector<int>& gets_input_from() const;
    virtual const vector<int>& sends_output_to() const;
    virtual void set_layers_inputs(vector<Matrix***> inputs_);
    virtual int get_vertical_stride();
    virtual int get_horizontal_stride();
};

class FullyConnected : public Layer {
    friend class Softmax;
    Matrix ***activation_input, ***output, ***output_derivative, ***output_error, ***output_error_helper, ***layers_delta;
    Matrix *removed_rows, *backup_weights, *backup_biases, ***backup_activation_input, ***backup_output, ***backup_output_derivative, ***backpup_output_error, ***backup_output_error_helper, ***backup_layers_delta;
    int neuron_type, outputlen, backup_outputlen, my_index;
    Layer **network_layers;
    bool dropout_happened;
    short int layer_type;
    Neuron neuron;
    Feature_map **fmap;
    int threadcount;
    virtual void destroy_dinamic_data();
    virtual void build_dinamic_data();
    vector<int> gets_input_from_, sends_output_to_;
    vector<Matrix***> inputs;
    virtual void set_layers_inputs(vector<Matrix***> inputs_);
    public:
    FullyConnected(int row, Layer **layers, vector<int> input_from, int neuron_type, int my_index);
    ~FullyConnected();
    virtual Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***next_layers_error, int threadindex);
    virtual void layers_output(Matrix **input, int threadindex);
    virtual Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex);
    virtual Matrix** derivate_layers_output(Matrix **input, int threadindex);
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    virtual void set_input(Matrix **input, int threadindex);
    virtual inline Matrix** get_output(int threadindex = 0);
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
    virtual void create_connections(vector<int> input_from, vector<int> output_to);
    virtual const vector<int>& gets_input_from() const;
    virtual const vector<int>& sends_output_to() const;
    virtual void set_graph_information(Layer **network_layers, int my_index);
    void set_threadcount(int threadcount, vector<Matrix***> inputs_);
    inline int get_threadcount();
};

class Softmax : public FullyConnected {
    public:
    Softmax(int row, Layer **layers, vector<int> input_from, int my_index);
    ~Softmax();
    Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***next_layers_error, int threadindex);
    void layers_output(Matrix **input, int threadindex);
    Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex);
    Matrix** derivate_layers_output(Matrix **input, int threadindex);
};


class InputLayer : public Layer {
    int threadcount;
    public:
    short int layer_type, next_layers_type;
    int outputlen, row, col, input_channel_count;
    Matrix ***outputs;
    Padding padd;
    InputLayer(int row, int col, int input_channel_count, int neuron_type, Padding &p, short int next_layers_type);
    ~InputLayer();
    Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***next_layers_error, int threadindex);
    void layers_output(Matrix **input, int threadindex);
    Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex);
    Matrix** derivate_layers_output(Matrix **input, int threadindex);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    void set_input(Matrix **input, int threadindex);
    inline Matrix** get_output(int threadindex = 0);
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
    void set_threadcount(int threadcount);
    inline int get_threadcount();
};

class Flatten : public Layer{
    Matrix ***output, ***layers_delta;
    vector<Matrix***> inputs;
    vector<int> gets_input_from_, sends_output_to_;
    Layer **network_layers;
    Feature_map **fmap;
    short int layer_type;
    int threadcount, outputlen, my_index, map_count;
    virtual void destroy_dinamic_data();
    virtual void build_dinamic_data();
    public:
    Flatten(vector<Matrix***> inputs, Layer **network_layers, int index, vector<int> input_from, int mapcout_);
    virtual ~Flatten();
    virtual Matrix** backpropagate(Matrix **input, Layer *next_layer, Feature_map **nabla, Matrix ***next_layers_error, int threadindex);
    virtual void layers_output(Matrix **input, int threadindex);
    virtual Matrix* get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex);
    virtual Matrix** derivate_layers_output(Matrix **input, int threadindex);
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer);
    virtual void set_input(Matrix **input, int threadindex);
    virtual inline Matrix** get_output(int threadindex = 0);
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
    virtual void set_threadcount(int threadcount, vector<Matrix***> inputs_);
    virtual inline int get_threadcount();
    virtual void create_connections(vector<int> input_from, vector<int> output_to);
    virtual const vector<int>& gets_input_from() const;
    virtual const vector<int>& sends_output_to() const;
    virtual void set_layers_inputs(vector<Matrix***> inputs_);
    virtual int get_vertical_stride();
    virtual int get_horizontal_stride();
};
#endif // LAYERS_H_INCLUDED
