#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "neurons/neuron.h"
#include "data_loader/MNIST_data.h"
#include "matrix/matrix.h"
#include "layers/layers.h"

#define STOCHASTIC 0
#define MOMENTUM 1
#define NESTEROV 2
#define RMSPROP 3

struct Accuracy{
    int correct_answers;
    double total_cost, execution_time;
};

class Network{
    int total_layers_num, layers_num, costfunction_type, input_row, input_col, input_channel_count;
    Layer **layers;
    LayerDescriptor **layerdsc;
    void construct_layers(LayerDescriptor **desc);
    inline void backpropagate(MNIST_data *training_data, Layers_features **nabla);
    inline void feedforward(Matrix **input);
    double cost(Matrix &required_output, int req_outp_indx, int test_data_len);
    void stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, double learning_rate, double regularization_rate);
    void momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum);
    void nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum);
    void rmsprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, Layers_features **nabla_momentum, Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator);
    void gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost, double regularization_rate,
                                 double denominator, double momentum, MNIST_data **test_data, int minibatch_count , int test_data_len,  int trainingdata_len);
    public:
    double dropout_probability;
    bool monitor_training_duration;
    void store(char *filename);
    void stochastic_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    void momentum_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    void nesterov_accelerated_gradient(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    void rmsprop(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, double denominator=0.00001, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int input_channel_count = 1,
            int costfunction_type = CROSS_ENTROPY_CF, double dropout_probability = 0);
    Network(char *data);
    ~Network();
    Matrix get_output(Matrix **input);
    Accuracy check_accuracy(MNIST_data **test_data, int test_data_len, int epoch, bool monitor_learning_cost = false, double regularization_rate = 0);
};

#endif // NETWORK_H_INCLUDED
