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
    virtual void stochastic(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, double learning_rate, double regularization_rate);
    virtual void momentum_based(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                        double learning_rate, double regularization_rate, double momentum);
    virtual void nesterov(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                  double learning_rate, double regularization_rate, double momentum);
    virtual void RMSprop(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                 Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator);
    virtual void gradient_descent_variant(int variant, Data_Loader **training_data, int epochs, int minibatch_len, double learning_rate, int change_learning_cost, double regularization_rate,
                                 double denominator, double momentum, Data_Loader **test_data, int minibatch_count , int test_data_len,  int trainingdata_len);
    public:
    double dropout_probability;
    bool monitor_training_duration;
    virtual void stochastic_gradient_descent(Data_Loader **training_data, int epochs, int minibatch_len, double learning_rate, int change_learning_cost = 0,
                                    double regularization_rate = 0, Data_Loader **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    virtual void momentum_gradient_descent(Data_Loader **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, int change_learning_cost = 0,
                                    double regularization_rate = 0, Data_Loader **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    virtual void nesterov_accelerated_gradient(Data_Loader **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, int change_learning_cost = 0,
                                    double regularization_rate = 0, Data_Loader **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    virtual void rmsprop(Data_Loader **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, int change_learning_cost = 0,
                                    double regularization_rate = 0, double denominator=0.00001, Data_Loader **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    StochasticGradientDescent(Network &neunet, int costfunction_type, double dropout_probability = 0);
    ~StochasticGradientDescent();
    Accuracy check_accuracy(Data_Loader **test_data, int test_data_len, int epoch, int change_learning_cost = 0, double regularization_rate = 0);
};

class StochasticGradientDescentMultiThread : public StochasticGradientDescent {
    ThreadPool tp;
    Job **job;
    public:
    StochasticGradientDescentMultiThread(Network &neunet, int costfunction_type, double dropout_probability = 0, int thread_count = 4);
    ~StochasticGradientDescentMultiThread();
    virtual void stochastic(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, double learning_rate, double regularization_rate);
    virtual void momentum_based(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                        double learning_rate, double regularization_rate, double momentum);
    virtual void nesterov(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                  double learning_rate, double regularization_rate, double momentum);
    virtual void RMSprop(Data_Loader **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla, Layers_features **nabla_momentum,
                 Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator);
    void gradient_descent_variant(int variant, Data_Loader **training_data, int epochs, int minibatch_len, double learning_rate, int change_learning_cost, double regularization_rate,
                                 double denominator, double momentum, Data_Loader **test_data, int minibatch_count , int test_data_len,  int trainingdata_len);

};


#endif // SGD_H_INCLUDED
