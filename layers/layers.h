#ifndef LAYERS_H_INCLUDED
#define LAYERS_H_INCLUDED

#include "../neuron.h"
#include "../MNIST_data.h"
#include "../matrice.h"

#define QUADRATIC_CF 0
#define CROSS_ENTROPY_CF 1

#define INPUTLAYER -1
#define FULLY_CONNECTED 0
#define CONVOLUTIONAL 1
#define POOLING 2
#define SOFTMAX 3

class LayerDescriptor{
    public:
    int layer_type;
    int neuron_count;
    int stride;
    int neuron_type;
    public:
    LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int stride = 1);
};

class Feature_map{
    void initialize_biases();
    void initialize_weights();
    int row, col;
    public:
    Matrice **weights, **biases;
    int mapdepth;
    Feature_map(int row, int col, int mapdepth, int biascnt = -1);
    int get_col();
    ~Feature_map();
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
    virtual inline void backpropagate(Matrice **input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error) = 0;
    virtual inline void layers_output(Matrice **input) = 0;
    virtual inline Matrice get_output_error(Matrice **input, Matrice &required_output, int costfunction_type) = 0;
    virtual inline Matrice derivate_layers_output(Matrice **input) = 0;
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases) = 0;
    virtual inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes) = 0;
    virtual inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes) = 0;
    virtual void set_input(Matrice **input) = 0;
    virtual inline Matrice** get_output() = 0;
    virtual inline Matrice* get_weights() = 0;
    virtual inline short get_layer_type() = 0;
    virtual inline int get_outputlen() = 0;
    virtual void set_weights(Matrice *w) = 0;
    virtual void set_biases(Matrice *b) = 0;
};

class FullyConnected : public Layer {
    Matrice **output;
    Feature_map fmap;
    int neuron_type, neuron_count, outputlen;
    short int layer_type;
    Neuron neuron;
    public:
    FullyConnected(int row, int prev_row, int neuron_type);
    ~FullyConnected();
    inline void backpropagate(Matrice **input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error);
    inline void layers_output(Matrice **input);
    inline Matrice get_output_error(Matrice **input, Matrice &required_output, int costfunction_type);
    inline Matrice derivate_layers_output(Matrice **input);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(Matrice **input);
    inline Matrice** get_output();
    inline Matrice* get_weights();
    inline short get_layer_type();
    inline int get_outputlen();
    void set_weights(Matrice *w);
    void set_biases(Matrice *b);
};

class Convolutional : public Layer {
    Matrice **outputs, **flattened_output;
    Feature_map **fmap;
    Padding pad;
    int neuron_type, outputlen, input_row, input_col, kernel_row, kernel_col, map_count, stride, next_layers_type, output_row, output_col;
    short int layer_type;
    Neuron neuron;
    public:
    Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int stride = 1);
    ~Convolutional();
    inline void backpropagate(Matrice **input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error);
    inline void layers_output(Matrice **input);
    inline Matrice get_output_error(Matrice **input, Matrice &required_output, int costfunction_type);
    inline Matrice derivate_layers_output(Matrice **input);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(Matrice **input);
    inline Matrice** get_output();
    inline Matrice* get_weights();
    inline short get_layer_type();
    inline int get_outputlen();
    void set_weights(Matrice *w);
    void set_biases(Matrice *b);
    void flatten();
    Matrice flatten_to_2D(Matrice &m);
};

/*class Pooling : public Layer {
    public:
    //Neuron neuron;
    Pooling(int row, int col, int neuron_type);
    ~Pooling();
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(double **input);
    inline Matrice& get_output();
    inline Matrice& get_weights();
    inline short get_layer_type();
    inline int get_outputlen();
};


class Softmax : public Layer {
    public:
    //Neuron neuron;
    Softmax(int row, int col, int neuron_type);
    ~Softmax();
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(double **input);
    inline Matrice& get_output();
    inline Matrice& get_weights();
    inline short get_layer_type();
    inline int get_outputlen();
};*/

class InputLayer : public Layer {
    public:
    short int layer_type, next_layers_type;
    int outputlen, row, col, feature_depth;
    Matrice **outputs;
    Padding padd;
    InputLayer(int row, int col, int feature_depth, int neuron_type, Padding &p, short int next_layers_type);
    ~InputLayer();
    inline void backpropagate(Matrice **input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &next_layers_error);
    inline void layers_output(Matrice **input);
    inline Matrice get_output_error(Matrice **input, Matrice &required_output, int costfunction_type);
    inline Matrice derivate_layers_output(Matrice **input);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(Matrice **input);
    inline Matrice** get_output();
    inline Matrice* get_weights();
    inline short get_layer_type();
    inline int get_outputlen();
    void set_weights(Matrice *w);
    void set_biases(Matrice *b);
};

#endif // LAYERS_H_INCLUDED
