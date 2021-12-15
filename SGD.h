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
    protected:
    int costfunction_type;
    Network &neunet;
    public:
    virtual double cost(Matrix &required_output, int req_outp_indx, int test_data_len);
    virtual void stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, double learning_rate, double regularization_rate);
    virtual void momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                        double learning_rate, double regularization_rate, double momentum);
    virtual void nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                  double learning_rate, double regularization_rate, double momentum);
    virtual void RMSprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                 Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator);
    virtual void gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost, double regularization_rate,
                                 double denominator, double momentum, MNIST_data **test_data, int minibatch_count , int test_data_len,  int trainingdata_len);
    public:
    double dropout_probability;
    bool monitor_training_duration;
    virtual void stochastic_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    virtual void momentum_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    virtual void nesterov_accelerated_gradient(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    virtual void rmsprop(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, double denominator=0.00001, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    StochasticGradientDescent(Network &neunet, int costfunction_type, double dropout_probability = 0);
    ~StochasticGradientDescent();
    Accuracy check_accuracy(MNIST_data **test_data, int test_data_len, int epoch, bool monitor_learning_cost = false, double regularization_rate = 0);
};

class StochasticGradientDescentMultiThread : public StochasticGradientDescent {
    ThreadPool tp;
    Job **job;
    public:
    StochasticGradientDescentMultiThread(Network &neunet, int costfunction_type, double dropout_probability = 0, int thread_count = 4);
    ~StochasticGradientDescentMultiThread();
    virtual void stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, double learning_rate, double regularization_rate);
    virtual void momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                        double learning_rate, double regularization_rate, double momentum);
    virtual void nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                  double learning_rate, double regularization_rate, double momentum);
    virtual void RMSprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                 Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator);
    void gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost, double regularization_rate,
                                 double denominator, double momentum, MNIST_data **test_data, int minibatch_count , int test_data_len,  int trainingdata_len);

};


#endif // SGD_H_INCLUDED
