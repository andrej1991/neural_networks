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
    LayerDescriptor(int layer_type, int neuron_count, int neuron_type, int stride = 1);
    inline int get_layer_type();
    inline int get_neuron_count();
    inline int get_stride();
    inline int get_neuron_type();
};

class Layer{
    public:
    Matrice biases, output, weights;
    int neuron_type, neuron_count, outputlen;
    Neuron neuron;
    short int layer_type;
    virtual ~Layer();
    void initialize_biases();
    void initialize_weights();
    virtual inline void layers_output(double **input, int inputlen) = 0;
    virtual inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type) = 0;
    virtual inline Matrice derivate_layers_output(double **input, int inputlen) = 0;
    virtual void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases) = 0;
    virtual inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes) = 0;
    virtual inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes) = 0;
    virtual void set_input(double **input, int input_len) = 0;
};

class FullyConnected : public Layer {
    public:
    //Neuron neuron;
    FullyConnected();
    ~FullyConnected();
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(double **input, int input_len);
};

class Convolutional : public Layer {
    public:
    //Neuron neuron;
    Convolutional();
    ~Convolutional();
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(double **input, int input_len);
};

class Pooling : public Layer {
    public:
    //Neuron neuron;
    Pooling();
    ~Pooling();
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(double **input, int input_len);
};


class Softmax : public Layer {
    public:
    //Neuron neuron;
    Softmax();
    ~Softmax();
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    void set_input(double **input, int input_len);
};

class InputLayer : public Layer {
    inline void layers_output(double **input, int layer);
    inline Matrice get_output_error(double **input, double **required_output, int inputlen, int costfunction_type);
    inline Matrice derivate_layers_output(double **input, int inputlen);
    void update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    public:
    InputLayer();
    ~InputLayer();
    void set_input(double **input, int input_len);
};

#endif // LAYERS_H_INCLUDED
