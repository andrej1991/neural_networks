#include <math.h>
#include <chrono>
#include <random>
#include "SGD.h"
#include "matrix/matrix.h"
#include "additional.h"


StochasticGradientDescent::StochasticGradientDescent(Network &neunet, int costfunction_type, double dropout_probability):
                                                    neunet(neunet), costfunction_type(costfunction_type), dropout_probability(dropout_probability){};

StochasticGradientDescent::~StochasticGradientDescent()
{
    ;
}

double StochasticGradientDescent::cost(Matrix &required_output, int req_outp_indx, int test_data_len)
{
    double helper = 0, result = 0;
    switch(this->costfunction_type)
    {
    case QUADRATIC_CF:
        /// 1/2 * ||y(x) - a||^2
        for(int i = 0; i < this->neunet.layers[this->neunet.layers_num - 1]->get_output_len(); i++)
        {
            helper = required_output.data[i][0] - this->neunet.layers[this->neunet.layers_num - 1]->get_output()[0]->data[i][0];
            result += helper * helper;
        }
        return (0.5 * result) / (test_data_len * 2.0);
    case CROSS_ENTROPY_CF:
        ///y(x)ln a + (1 - y(x))ln(1 - a)
        for(int i = 0; i < this->neunet.layers[this->neunet.layers_num - 1]->get_output_len(); i++)
        {
            helper += required_output.data[i][0] * log(this->neunet.layers[this->neunet.layers_num - 1]->get_output()[0]->data[i][0]) + (1 - required_output.data[i][0]) *
                            log(1 - this->neunet.layers[this->neunet.layers_num - 1]->get_output()[0]->data[i][0]);
        }
        return helper / test_data_len;
    case LOG_LIKELIHOOD_CF:
        result = -1 * log(this->neunet.layers[this->neunet.layers_num - 1]->get_output()[0]->data[req_outp_indx][0]);
        return result / test_data_len;
    default:
        cerr << "Unknown cost function\n";
        throw exception();
    }
}

void StochasticGradientDescent::stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                           double learning_rate, double regularization_rate)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->neunet.backpropagate(minibatches[training_data_index], deltanabla[0], this->costfunction_type, 0);
        for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[0][layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        this->neunet.layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, nabla[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescent::momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                               Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->neunet.backpropagate(minibatches[training_data_index], deltanabla[0], this->costfunction_type, 0);
        for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[0][layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        nabla_momentum[layer_index][0] = (nabla_momentum[layer_index][0] * momentum) + (nabla[layer_index][0]*(1 - momentum));
        this->neunet.layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, nabla_momentum[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescent::nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                         Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum)
{
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        this->neunet.layers[layer_index]->update_weights_and_biasses(momentum, regularization_rate, nabla_momentum[layer_index]);
    }
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->neunet.backpropagate(minibatches[training_data_index], deltanabla[0], this->costfunction_type, 0);
        for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[0][layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        this->neunet.layers[layer_index]->update_weights_and_biasses(-1*momentum, regularization_rate, nabla_momentum[layer_index]);
        nabla_momentum[layer_index][0] = (nabla_momentum[layer_index][0] * momentum) + (nabla[layer_index][0] * learning_rate);
        this->neunet.layers[layer_index]->update_weights_and_biasses(1, regularization_rate, nabla_momentum[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescent::RMSprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                        Layers_features **squared_grad_moving_avarange, Layers_features **layer_helper, double learning_rate,
                                        double regularization_rate, double momentum, double denominator)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->neunet.backpropagate(minibatches[training_data_index], deltanabla[0], this->costfunction_type, 0);
        for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[0][layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        squared_grad_moving_avarange[layer_index][0] = (squared_grad_moving_avarange[layer_index][0] * momentum) + (nabla[layer_index][0].square_element_by() * (1 - momentum));
        layer_helper[layer_index][0] = nabla[layer_index][0] / (squared_grad_moving_avarange[layer_index][0].sqroot() + denominator);
        this->neunet.layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, layer_helper[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescent::gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost,
                                                         double regularization_rate, double denominator, double momentum, MNIST_data **test_data, int minibatch_count,
                                                         int test_data_len,  int trainingdata_len)
{

    if(minibatch_count < 0)
    {
        minibatch_count = trainingdata_len / minibatch_len;
    }
    Accuracy execution_accuracy;
    MNIST_data *minibatches[minibatch_count][minibatch_len];
    std::random_device rand;
    std::uniform_int_distribution<int> distribution(0, trainingdata_len-1);
    int learnig_cost_counter = 0;
    int biasrow, biascol;
    double previoius_learning_cost = 0;
    double lr, reg;
    Matrix helper(this->neunet.layers[this->neunet.layers_num - 1]->get_output_row(), 1);
    Matrix dropout_neurons[this->neunet.layers_num - 1];
    chrono::time_point<chrono::system_clock> start, end_training;
    chrono::duration<double> epoch_duration, overall_duration;
    Layers_features **nabla, ***deltanabla, **helper_1, **helper_2;
    try
    {
        deltanabla = new Layers_features** [minibatch_len];
        nabla = new Layers_features* [this->neunet.layers_num];
        for(int i = 0; i < minibatch_len; i++)
        {
            deltanabla[i] = new Layers_features* [this->neunet.layers_num];
        }
        helper_1 = new Layers_features* [this->neunet.layers_num];
        helper_2 = new Layers_features* [this->neunet.layers_num];
        for(int i = 0; i < this->neunet.layers_num; i++)
        {
            ///Layers_features(int mapcount, int row, int col, int depth, int biascnt);
            biasrow = this->neunet.layers[i]->get_output_row();
            biascol = this->neunet.layers[i]->get_output_col();
            nabla[i] = new Layers_features(this->neunet.layers[i]->get_mapcount(),
                                           this->neunet.layers[i]->get_weights_row(),
                                           this->neunet.layers[i]->get_weights_col(),
                                           this->neunet.layers[i]->get_mapdepth(),
                                           biasrow, biascol);
            nabla[i]->zero();
            for(int j = 0; j < minibatch_len; j++)
            {
                deltanabla[j][i] = new Layers_features(this->neunet.layers[i]->get_mapcount(),
                                                    this->neunet.layers[i]->get_weights_row(),
                                                    this->neunet.layers[i]->get_weights_col(),
                                                    this->neunet.layers[i]->get_mapdepth(),
                                                    biasrow, biascol);
            }
            helper_1[i] = new Layers_features(this->neunet.layers[i]->get_mapcount(),
                                                this->neunet.layers[i]->get_weights_row(),
                                                this->neunet.layers[i]->get_weights_col(),
                                                this->neunet.layers[i]->get_mapdepth(),
                                                biasrow, biascol);
            helper_1[i]->zero();
            helper_2[i] = new Layers_features(this->neunet.layers[i]->get_mapcount(),
                                                this->neunet.layers[i]->get_weights_row(),
                                                this->neunet.layers[i]->get_weights_col(),
                                                this->neunet.layers[i]->get_mapdepth(),
                                                biasrow, biascol);
        }
    }
    catch(bad_alloc& ba)
    {
        cerr<<"operator new failed in the function: Network::update_weights_and_biasses"<<endl;
        return;
    }
    helper.zero();
    Matrix output;
    for(int i = 0; i < epochs; i++)
    {
        for(int j = 0; j < minibatch_count; j++)
        {
            for(int k = 0; k < minibatch_len; k++)
            {
                minibatches[j][k] = training_data[distribution(rand)];
            }
        }
        start = chrono::system_clock::now();
        for(int j = 0; j < minibatch_count; j++)
        {
            if(this->dropout_probability != 0)
            {
                int dropout_index = 0;
                while(this->neunet.layers[dropout_index]->get_layer_type() != FULLY_CONNECTED and this->neunet.layers[dropout_index]->get_layer_type() != SOFTMAX)
                {
                    dropout_index++;
                }
                dropout_neurons[dropout_index] = this->neunet.layers[dropout_index]->drop_out_some_neurons(dropout_probability, NULL);
                dropout_index++;
                for(dropout_index; dropout_index < this->neunet.layers_num - 1; dropout_index++)
                {
                    dropout_neurons[dropout_index] = this->neunet.layers[dropout_index]->drop_out_some_neurons(dropout_probability, &dropout_neurons[dropout_index-1]);
                }
                this->neunet.layers[this->neunet.layers_num - 1]->drop_out_some_neurons(0.0, &dropout_neurons[this->neunet.layers_num - 2]);
                for(int layerindex = 0; layerindex < this->neunet.layers_num; layerindex++)
                {
                    if(this->neunet.layers[layerindex]->get_layer_type() == FULLY_CONNECTED or this->neunet.layers[layerindex]->get_layer_type() == SOFTMAX)
                    {
                        delete nabla[layerindex];
                        delete helper_1[layerindex];
                        delete helper_2[layerindex];

                        biasrow = this->neunet.layers[layerindex]->get_output_row();
                        biascol = this->neunet.layers[layerindex]->get_output_col();
                        nabla[layerindex] = new Layers_features(this->neunet.layers[layerindex]->get_mapcount(),
                                                       this->neunet.layers[layerindex]->get_weights_row(),
                                                       this->neunet.layers[layerindex]->get_weights_col(),
                                                       this->neunet.layers[layerindex]->get_mapdepth(),
                                                       biasrow, biascol);
                        helper_1[layerindex] = new Layers_features(this->neunet.layers[layerindex]->get_mapcount(),
                                                            this->neunet.layers[layerindex]->get_weights_row(),
                                                            this->neunet.layers[layerindex]->get_weights_col(),
                                                            this->neunet.layers[layerindex]->get_mapdepth(),
                                                            biasrow, biascol);
                        helper_2[layerindex] = new Layers_features(this->neunet.layers[layerindex]->get_mapcount(),
                                                            this->neunet.layers[layerindex]->get_weights_row(),
                                                            this->neunet.layers[layerindex]->get_weights_col(),
                                                            this->neunet.layers[layerindex]->get_mapdepth(),
                                                            biasrow, biascol);
                        for(int jobindex = 0; jobindex < minibatch_len; jobindex++)
                        {
                            delete deltanabla[jobindex][layerindex];
                            deltanabla[jobindex][layerindex] = new Layers_features(this->neunet.layers[layerindex]->get_mapcount(),
                                                                                   this->neunet.layers[layerindex]->get_weights_row(),
                                                                                   this->neunet.layers[layerindex]->get_weights_col(),
                                                                                   this->neunet.layers[layerindex]->get_mapdepth(),
                                                                                   biasrow, biascol);
                        }
                    }
                }
            }
            lr = learning_rate / minibatch_len;
            reg = (1 - learning_rate * (regularization_rate / trainingdata_len));
            switch(variant)
            {
                case STOCHASTIC:
                    this->stochastic(minibatches[j], minibatch_len, nabla, deltanabla, lr, reg);
                    break;
                case MOMENTUM:
                    this->momentum_based(minibatches[j], minibatch_len, nabla, deltanabla, helper_1, lr, reg, momentum);
                    break;
                case NESTEROV:
                    this->nesterov(minibatches[j], minibatch_len, nabla, deltanabla, helper_1, lr, reg, momentum);
                    break;
                case RMSPROP:
                    this->RMSprop(minibatches[j], minibatch_len, nabla, deltanabla, helper_1, helper_2, lr, reg, momentum, denominator);
                    break;
                default:
                    throw invalid_argument("Unknown gradient descent variant!");
            }
            for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
            {
                nabla[layer_index]->zero();
            }
            if(this->dropout_probability != 0)
            {
                int dropout_index = 0;
                while(this->neunet.layers[dropout_index]->get_layer_type() != FULLY_CONNECTED and this->neunet.layers[dropout_index]->get_layer_type() != SOFTMAX)
                {
                    dropout_index++;
                }
                this->neunet.layers[dropout_index]->restore_neurons(NULL);
                dropout_index++;
                for(dropout_index; dropout_index < this->neunet.layers_num - 1; dropout_index++)
                {
                    this->neunet.layers[dropout_index]->restore_neurons(&dropout_neurons[dropout_index-1]);
                }
                this->neunet.layers[this->neunet.layers_num - 1]->restore_neurons(&dropout_neurons[this->neunet.layers_num - 2]);
            }
        }
        for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
        {
            helper_1[layer_index]->zero();
        }
        end_training = chrono::system_clock::now();
        epoch_duration = end_training - start;
        if(test_data != NULL)
        {
            execution_accuracy = this->check_accuracy(test_data, test_data_len, i, monitor_learning_cost, regularization_rate);
            if(monitor_learning_cost)
            {
                cout << "total cost: " << execution_accuracy.total_cost << endl;
                if(abs((long long int)execution_accuracy.total_cost) > abs((long long int)previoius_learning_cost))
                    learnig_cost_counter++;
                if(learnig_cost_counter == 10)
                {
                    learnig_cost_counter = 0;
                    learning_rate == 0 ? learning_rate = 1 : learning_rate /= 2.0;
                    cout << "changing leatning rate to: " << learning_rate << endl;
                }
                previoius_learning_cost = execution_accuracy.total_cost;
            }
        }
        if(this->monitor_training_duration)
        {
            cout << "    training over an epoch took: " << epoch_duration.count() << "seconds" << endl;
            if(test_data != NULL)
            {
                end_training = chrono::system_clock::now();
                overall_duration = end_training - start;
                cout << "    the testing took: " << execution_accuracy.execution_time << "seconds" << endl;
                cout << "    the whole epoch took: " << overall_duration.count() << "seconds" << endl;
            }
        }
    }
    for(int i = 0; i < minibatch_len; i++)
    {
        for(int j = 0; j < this->neunet.layers_num; j++)
        {
            delete deltanabla[i][j];
        }
    }
    for(int i = 0; i < this->neunet.layers_num; i++)
    {
        delete nabla[i];
        //delete[] deltanabla[i];
        delete helper_1[i];
        delete helper_2[i];
    }
    delete[] nabla;
    delete[] deltanabla;
    delete[] helper_1;
    delete[] helper_2;
}

void StochasticGradientDescent::stochastic_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    this->gradient_descent_variant(STOCHASTIC, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 0, 0, test_data, minibatch_count, test_data_len, trainingdata_len);
}

void StochasticGradientDescent::momentum_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    /*if(this->dropout_probability != 0.0)
    {
        throw invalid_argument("Known limitation that momentum_gradient_descent is not working with dropout.");
    }*/
    this->gradient_descent_variant(MOMENTUM, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 0, momentum, test_data, minibatch_count, test_data_len, trainingdata_len);
}

void StochasticGradientDescent::nesterov_accelerated_gradient(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    if(this->dropout_probability != 0.0)
    {
        throw invalid_argument("Known limitation that nesterov_accelerated_gradient is not working with dropout.");
    }
    this->gradient_descent_variant(NESTEROV, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 0, momentum, test_data, minibatch_count, test_data_len, trainingdata_len);
}

void StochasticGradientDescent::rmsprop(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost,
                                            double regularization_rate, double denominator, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    /*if(this->dropout_probability != 0.0)
    {
        throw invalid_argument("Known limitation that rmsprop is not working with dropout.");
    }*/
    this->gradient_descent_variant(RMSPROP, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 denominator, momentum, test_data, minibatch_count, test_data_len, trainingdata_len);
}

Accuracy StochasticGradientDescent::check_accuracy(MNIST_data **test_data, int test_data_len, int epoch, bool monitor_learning_cost, double regularization_rate)
{
    int learning_accuracy;
    double learning_cost, squared_sum = 0, avarange_confidence = 0;
    Matrix helper(this->neunet.layers[this->neunet.layers_num - 1]->get_output_row(), 1);
    Matrix output;
    int mapcount, mapdepth;
    Feature_map **fmaps;
    chrono::time_point<chrono::system_clock> start, end_testing;
    chrono::duration<double> test_duration;
    for(int i = 0; i < this->neunet.layers[this->neunet.layers_num - 1]->get_output_row(); i++)
    {
        helper.data[i][0] = 0;
    }
    learning_accuracy = learning_cost = 0;
    start = chrono::system_clock::now();
    for(int j = 0; j < test_data_len; j++)
    {
        ///TODO this is an errorprone as well
        output = this->neunet.get_output(test_data[j]->input);
        if(getmax(output.data, output.get_row()) == test_data[j]->required_output.data[0][0])
        {
            learning_accuracy++;
            avarange_confidence += output.data[int(test_data[j]->required_output.data[0][0])][0] * 100.0;
        }
        if(monitor_learning_cost)
        {
            helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 1;
            learning_cost += this->cost(helper, test_data[j]->required_output.data[0][0], test_data_len);
            helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 0;
        }
    }
    if((regularization_rate != 0) and (monitor_learning_cost))
    {
        for(int layerindex = 0; layerindex < this->neunet.layers_num; layerindex++)
        {
            if(this->neunet.layers[layerindex]->get_layer_type() != POOLING)
            {
                fmaps = this->neunet.layers[layerindex]->get_feature_maps();
                mapcount = this->neunet.layers[layerindex]->get_mapcount();
                for(int mapindex = 0; mapindex < mapcount; mapindex++)
                {
                    mapdepth = fmaps[mapindex]->get_mapdepth();
                    for(int i = 0; i < mapdepth; i++)
                    {
                        squared_sum += fmaps[mapindex]->weights[i]->squared_sum_over_elements();
                    }
                }
            }
        }
        learning_cost += ((regularization_rate/(2.0*test_data_len))*squared_sum);
        squared_sum = 0;
    }
    end_testing = chrono::system_clock::now();
    test_duration = end_testing - start;
    cout << "set " << epoch << ": " << learning_accuracy << " out of: " << test_data_len << endl;
    cout << "  The avarange confidence is: " << avarange_confidence/learning_accuracy << "%" << endl;
    double execution_time = 0;
    if(this->monitor_training_duration)
    {
        execution_time = test_duration.count();
    }
    return Accuracy {learning_accuracy, learning_cost, execution_time};
}


StochasticGradientDescentMultiThread::StochasticGradientDescentMultiThread(Network &neunet, int costfunction_type, double dropout_probability, int thread_count):
                                                    tp(thread_count), StochasticGradientDescent(neunet, costfunction_type, dropout_probability){};


StochasticGradientDescentMultiThread::~StochasticGradientDescentMultiThread()
{
    ;
}

void StochasticGradientDescentMultiThread::stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                           double learning_rate, double regularization_rate)
{
    //cout << "stochastic is being called\n";
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->job[training_data_index]->costfunction_type = this->costfunction_type;
        this->job[training_data_index]->training_data = minibatches[training_data_index];
        this->job[training_data_index]->deltanabla = deltanabla[training_data_index];
        this->job[training_data_index]->nabla = nabla;
        this->tp.push(this->job[training_data_index]);
    }
    this->tp.wait();
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        this->neunet.layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, nabla[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescentMultiThread::momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                               Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->job[training_data_index]->costfunction_type = this->costfunction_type;
        this->job[training_data_index]->training_data = minibatches[training_data_index];
        this->job[training_data_index]->deltanabla = deltanabla[training_data_index];
        this->job[training_data_index]->nabla = nabla;
        this->tp.push(this->job[training_data_index]);
    }
    this->tp.wait();
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        nabla_momentum[layer_index][0] = (nabla_momentum[layer_index][0] * momentum) + (nabla[layer_index][0]*(1 - momentum));
        this->neunet.layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, nabla_momentum[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescentMultiThread::nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                         Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum)
{
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        this->neunet.layers[layer_index]->update_weights_and_biasses(momentum, regularization_rate, nabla_momentum[layer_index]);
    }
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->job[training_data_index]->costfunction_type = this->costfunction_type;
        this->job[training_data_index]->training_data = minibatches[training_data_index];
        this->job[training_data_index]->deltanabla = deltanabla[training_data_index];
        this->job[training_data_index]->nabla = nabla;
        this->tp.push(this->job[training_data_index]);
    }
    this->tp.wait();
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        this->neunet.layers[layer_index]->update_weights_and_biasses(-1*momentum, regularization_rate, nabla_momentum[layer_index]);
        nabla_momentum[layer_index][0] = (nabla_momentum[layer_index][0] * momentum) + (nabla[layer_index][0] * learning_rate);
        this->neunet.layers[layer_index]->update_weights_and_biasses(1, regularization_rate, nabla_momentum[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescentMultiThread::RMSprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features ***deltanabla,
                                        Layers_features **squared_grad_moving_avarange, Layers_features **layer_helper, double learning_rate,
                                        double regularization_rate, double momentum, double denominator)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->job[training_data_index]->costfunction_type = this->costfunction_type;
        this->job[training_data_index]->training_data = minibatches[training_data_index];
        this->job[training_data_index]->deltanabla = deltanabla[training_data_index];
        this->job[training_data_index]->nabla = nabla;
        this->tp.push(this->job[training_data_index]);
    }
    this->tp.wait();
    for(int layer_index = 0; layer_index < this->neunet.layers_num; layer_index++)
    {
        squared_grad_moving_avarange[layer_index][0] = (squared_grad_moving_avarange[layer_index][0] * momentum) + (nabla[layer_index][0].square_element_by() * (1 - momentum));
        layer_helper[layer_index][0] = nabla[layer_index][0] / (squared_grad_moving_avarange[layer_index][0].sqroot() + denominator);
        this->neunet.layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, layer_helper[layer_index]);
        nabla[layer_index]->zero();
    }
}

void StochasticGradientDescentMultiThread::gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost,
                                                         double regularization_rate, double denominator, double momentum, MNIST_data **test_data, int minibatch_count,
                                                         int test_data_len,  int trainingdata_len)
{
    //cout << "multithreaded SGD variant is being called\n";
    this->job = new Job* [minibatch_len];
    for(int i = 0; i < minibatch_len; i++)
        this->job[i] = new Job(i, &(this->neunet));
    int prev_tc = this->neunet.get_threadcount();
    this->neunet.set_threadcount(minibatch_len);
    StochasticGradientDescent::gradient_descent_variant(variant, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost,
                                                         regularization_rate, denominator, momentum, test_data, minibatch_count,
                                                         test_data_len, trainingdata_len);
    this->neunet.set_threadcount(prev_tc);
}
