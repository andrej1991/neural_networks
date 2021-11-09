#ifndef SGD_H_INCLUDED
#define SGD_H_INCLUDED

#include "network.h"
#include "multithreading/threadpool.h"

#define STOCHASTIC 0
#define MOMENTUM 1
#define NESTEROV 2
#define RMSPROP 3

struct Accuracy{
    int correct_answers;
    double total_cost, execution_time;
};

class StochasticGradientDescent{
    int costfunction_type;
    Network &neunet;
    ThreadPool tp;
    Job **job;
    public:
    double cost(Matrix &required_output, int req_outp_indx, int test_data_len);
    void stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, double learning_rate, double regularization_rate);
    void momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum);
    void nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum);
    void rmsprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum, Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator);
    void gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost, double regularization_rate,
                                 double denominator, double momentum, MNIST_data **test_data, int minibatch_count , int test_data_len,  int trainingdata_len);
    public:
    double dropout_probability;
    bool monitor_training_duration;
    void stochastic_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    void momentum_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    void nesterov_accelerated_gradient(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    void rmsprop(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, double denominator=0.00001, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    StochasticGradientDescent(Network &neunet, int costfunction_type, double dropout_probability = 0, int thread_count = 4);
    ~StochasticGradientDescent();
    Accuracy check_accuracy(MNIST_data **test_data, int test_data_len, int epoch, bool monitor_learning_cost = false, double regularization_rate = 0);
};


#endif // SGD_H_INCLUDED
