#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED
#include "neuron.h"
#include "MNIST_data.h"
#include "matrix.h"
#include "layers/layers.h"

struct Accuracy{
    int correct_answers;
    double total_cost, execution_time;
};

class Network{
    int total_layers_num, layers_num, costfunction_type, input_row, input_col, input_channel_count;
    Layer **layers;
    LayerDescriptor **layerdsc;
    bool dropout;
    void construct_layers(LayerDescriptor **desc);
    inline void backpropagate(MNIST_data *training_data, Layers_features **nabla);
    inline void remove_some_neurons(Matrix ***w_bckup, Matrix ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrix **w_bckup, Matrix **b_bckup, int *layers_bckup, int **indexes);
    inline void feedforward(Matrix **input);
    double cost(Matrix &required_output, int req_outp_indx);
    public:
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
            int costfunction_type = CROSS_ENTROPY_CF, bool dropout = false);
    Network(char *data);
    ~Network();
    void test(MNIST_data **d, MNIST_data **v);
    Matrix get_output(Matrix **input);
    Accuracy check_accuracy(MNIST_data **test_data, int test_data_len, int epoch, bool monitor_learning_cost = false);
};

#endif // NETWORK_H_INCLUDED
