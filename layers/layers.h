#ifndef LAYERS_H_INCLUDED
#define LAYERS_H_INCLUDED

#include <iostream>

#include "../neuron/neuron.h"
#include "../MNIST_data.h"
#include "../matrix/matrix.h"
#include "../opencl_setup.h"

#define QUADRATIC_CF 0
#define CROSS_ENTROPY_CF 1
#define LOG_LIKELIHOOD_CF 2

#define INPUTLAYER -1
#define FULLY_CONNECTED 0
#define CONVOLUTIONAL 1
#define POOLING 2
#define SOFTMAX 3

class LayerDescriptor{
    public:
    int layer_type;
    int neuron_count, row, col;
    int mapcount;
    int stride;
    int neuron_type;
    public:
    LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1);
};

class Feature_map{
    void initialize_biases(cl_context *context);
    void initialize_weights(cl_context *context);
    int row, col, mapdepth;
    public:
    MatrixOperations *mtxop;
    MatrixData **weights, **biases;
    OpenclSetup *openclenv;
    Feature_map(int row, int col, int mapdepth, int biascnt = -1, OpenclSetup *env=NULL, bool initialization_needed=true);
    int get_col();
    int get_row();
    int get_mapdepth();
    void store(std::ofstream &params);
    void load(std::ifstream &params);
    ~Feature_map();
};

class Layers_features{
    int fmap_count;
    public:
    Feature_map **fmap;
    Layers_features(int mapcount, int row, int col, int depth, int biascnt, OpenclSetup *env=NULL);
    ~Layers_features();
    void operator+=(Layers_features &layer);
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
    virtual inline MatrixData** backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **next_layers_error, int next_layers_fmapcount) = 0;
    virtual inline void layers_output(MatrixData **input) = 0;
    virtual void sync_memory() = 0;
    virtual inline MatrixData** get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type) = 0;
    virtual inline MatrixData** derivate_layers_output(MatrixData **input) = 0;
    virtual void update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer) = 0;
    virtual inline void remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes) = 0;
    virtual inline void add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes) = 0;
    virtual void set_input(MatrixData **input) = 0;
    virtual inline MatrixData** get_output() = 0;
    virtual inline Feature_map** get_feature_maps() = 0;
    virtual inline short get_layer_type() = 0;
    virtual inline int get_output_len() = 0;
    virtual inline int get_output_row() = 0;
    virtual inline int get_output_col() = 0;
    virtual void set_weights(MatrixData *w) = 0;
    virtual void set_biases(MatrixData *b) = 0;
    virtual int get_mapcount() = 0;
    virtual int get_mapdepth() = 0;
    virtual int get_weights_row() = 0;
    virtual int get_weights_col() = 0;
    virtual void store(std::ofstream &params) = 0;
    virtual void load(std::ifstream &params) = 0;

};

#define FullyConnectedFuncVarCount 5

class FullyConnected : public Layer {
    friend class Softmax;
    OpenclSetup *env;
    MatrixData **output, **output_error, **output_derivative, *function_variables[FullyConnectedFuncVarCount];
    int neuron_type, neuron_count, outputlen;
    short int layer_type;
    Neuron neuron;
    Feature_map **fmap;
    cl_program fully_connected_program;
    cl_kernel update_weights_kernel, update_biases_kernel, get_act_input_kernel, get_layers_delta_kernel, out_err_quad_cf_kernel;
    public:
    FullyConnected(int row, int prev_row, int neuron_type, OpenclSetup *env);
    ~FullyConnected();
    virtual inline MatrixData** backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **next_layers_error, int next_layers_fmapcount);
    virtual inline void layers_output(MatrixData **input);
    virtual void sync_memory();
    virtual inline MatrixData** get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type);
    virtual inline MatrixData** derivate_layers_output(MatrixData **input);
    virtual void update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer);
    virtual inline void remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes);
    virtual inline void add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes);
    virtual void set_input(MatrixData **input);
    virtual inline MatrixData** get_output();
    virtual inline Feature_map** get_feature_maps();
    virtual inline short get_layer_type();
    virtual inline int get_output_len();
    virtual inline int get_output_row();
    virtual inline int get_output_col();
    virtual void set_weights(MatrixData *w);
    virtual void set_biases(MatrixData *b);
    virtual int get_mapcount();
    virtual int get_mapdepth();
    virtual int get_weights_row();
    virtual int get_weights_col();
    virtual void store(std::ofstream &params);
    virtual void load(std::ifstream &params);
};

class Softmax : public FullyConnected {
    cl_program softmax_program;
    cl_kernel softmax_kernel, softmax_derivative_kernel, softmax_helper_kernel;
    public:
    Softmax(int row, int col, OpenclSetup *env);
    ~Softmax();
    inline MatrixData** backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **next_layers_error, int next_layers_fmapcount);
    inline void layers_output(MatrixData **input);
    inline MatrixData** get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type);
    inline MatrixData** derivate_layers_output(MatrixData **input);
};

class Convolutional : public Layer {
    MatrixData **outputs, **flattened_output, **convolution_helper, **output_derivative, **layers_delta, **layers_delta_helper, *_2Dkernel;
    cl_mem delta_helper;
    Feature_map **fmap;
    Padding pad;
    int neuron_type, outputlen, input_row, input_col, kernel_row, kernel_col, map_count, stride, next_layers_type, output_row, output_col;
    short int layer_type;
    Neuron neuron;
    OpenclSetup *env;
    cl_program convolutional_program, general_program;
    cl_kernel conv_and_add_kernel, fullconv_and_add_kernel, update_weights_kernel;
    inline void fulldepth_conv(MatrixData **input, cl_kernel *opencl_kernel);
    public:
    Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, OpenclSetup *env, int stride = 1);
    ~Convolutional();
    inline MatrixData** backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **next_layers_error, int next_layers_fmapcount);
    inline void layers_output(MatrixData **input);
    void sync_memory();
    inline MatrixData** get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type);
    inline MatrixData** derivate_layers_output(MatrixData **input);
    void update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer);
    inline void remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes);
    void set_input(MatrixData **input);
    inline MatrixData** get_output();
    inline Feature_map** get_feature_maps();
    inline short get_layer_type();
    inline int get_output_len();
    inline int get_output_row();
    inline int get_output_col();
    void set_weights(MatrixData *w);
    void set_biases(MatrixData *b);
    void flatten();
    int get_mapcount();
    int get_mapdepth();
    int get_weights_row();
    int get_weights_col();
    void get_2D_weights(int neuron_id, int fmap_id, MatrixData &kernel, Feature_map **next_layers_fmap);
    void store(std::ofstream &params);
    void load(std::ifstream &params);
};


class InputLayer : public Layer {
    OpenclSetup *env;
    cl_command_queue q;
    public:
    short int layer_type, next_layers_type;
    int outputlen, row, col, input_channel_count;
    MatrixData **outputs;
    Padding padd;
    InputLayer(int row, int col, int input_channel_count, int neuron_type, Padding &p, short int next_layers_type, OpenclSetup *env);
    ~InputLayer();
    inline MatrixData** backpropagate(MatrixData **input, Feature_map** next_layers_fmaps, Feature_map** nabla, MatrixData **next_layers_error, int next_layers_fmapcount);
    inline void layers_output(MatrixData **input);
    void sync_memory();
    inline MatrixData** get_output_error(MatrixData **input, MatrixData &required_output, int costfunction_type);
    inline MatrixData** derivate_layers_output(MatrixData **input);
    void update_weights_and_biasses(float learning_rate, float regularization_rate, Layers_features *layer);
    inline void remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes);
    void set_input(MatrixData **input);
    inline MatrixData** get_output();
    inline Feature_map** get_feature_maps();
    inline short get_layer_type();
    inline int get_output_len();
    inline int get_output_row();
    inline int get_output_col();
    void set_weights(MatrixData *w);
    void set_biases(MatrixData *b);
    int get_mapcount();
    int get_mapdepth();
    int get_weights_row();
    int get_weights_col();
    void store(std::ofstream &params);
    void load(std::ifstream &params);
};

/*class Pooling : public Layer {
    public:
    //Neuron neuron;
    Pooling(int row, int col, int neuron_type, OpenclSetup *env);
    ~Pooling();
    inline void layers_output(float **input, int layer);
    inline Matrice get_output_error(float **input, float **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(float **input, int inputlen);
    void update_weights_and_biasses(float learning_rate, float regularization_rate, Matrice *weights, Matrice *biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(float **input);
    inline Matrice& get_output();
    inline Matrice& get_weights();
    inline short get_layer_type();
    inline int get_outputlen();
};*/

#endif // LAYERS_H_INCLUDED
