#include "network.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <random>
#include "additional.h"
#include <math.h>
#include "layers/layers.h"
#include <chrono>

using namespace std;
///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int costfunction_type = CROSS_ENTROPY_CF, bool dropout = false
Network::Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  double dropout_probability):
                dropout_probability(dropout_probability), input_row(input_row), input_col(input_col), input_channel_count(input_channel_count), layers_num(layers_num), monitor_training_duration(true)
{
    try
    {
        this->costfunction_type = costfunction_type;
        this->total_layers_num = layers_num + 1;
        this->construct_layers(layerdesc);
    }
    catch(bad_alloc &ba)
        {
            cerr << "bad alloc in Network::constructor" << endl;
        }
}

Network::Network(char *data): monitor_training_duration(false)
{
    ///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout
    ///int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1
    ifstream file (data, ios::in|ios::binary);
    if(file.is_open())
        {
            int f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_vertical_stride, f_horizontal_stride;
            file.read(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_col )), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->costfunction_type)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->dropout_probability)), sizeof(double));
            this->total_layers_num = this->layers_num + 1;
            LayerDescriptor *dsc[this->layers_num];
            for(int i = 0; i < this->layers_num; i++)
                {
                    file.read(reinterpret_cast<char *>(&f_layer_type), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_neuron_type), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_neuron_count), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_col), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_mapcount), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_vertical_stride), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_horizontal_stride), sizeof(int));
                    dsc[i] = new LayerDescriptor(f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_vertical_stride, f_horizontal_stride);
                }
            this->construct_layers(dsc);
            for(int i = 0; i < this->layers_num; i++)
                {
                    this->layers[i]->load(file);
                }
            file.close();
            for(int i = 0; i < this->layers_num; i++)
                {
                    delete dsc[i];
                }
        }
    else
        {
            cerr << "Unable to open the file:" << '"' << data << '"' << endl;
            throw exception();
        }
}

Network::~Network()
{
    this->layers -= 1;
    for(int i = 0; i < this->layers_num; i++)
        {
            delete this->layers[i];
            delete this->layerdsc[i];
        }
    delete this->layers[this->layers_num];
    delete[] this->layers;
    delete[] this->layerdsc;
}

void Network::construct_layers(LayerDescriptor **layerdesc)
{
    this->layers = new Layer* [this->total_layers_num];
    this->layerdsc = new LayerDescriptor* [this->layers_num];
    Padding p;
    if(layerdesc[0]->layer_type == FULLY_CONNECTED)
        this->layers[0] = new InputLayer(input_row, 1, 1, SIGMOID, p, FULLY_CONNECTED);
    else
        this->layers[0] = new InputLayer(input_row, input_col, input_channel_count, SIGMOID, p, CONVOLUTIONAL);
    this->layers += 1;
    for(int i = 0; i < layers_num; i++)
    {
        this->layerdsc[i] = new LayerDescriptor(layerdesc[i]->layer_type, layerdesc[i]->neuron_type, layerdesc[i]->neuron_count,
                                                layerdesc[i]->col, layerdesc[i]->mapcount, layerdesc[i]->vertical_stride, layerdesc[i]->horizontal_stride);
        switch(layerdesc[i]->layer_type)
        {
            case FULLY_CONNECTED:
                this->layers[i] = new FullyConnected(layerdesc[i]->neuron_count, this->layers[i - 1]->get_output_len(),
                                                     layerdesc[i]->neuron_type);
                break;
            case SOFTMAX:
                this->layers[i] = new Softmax(layerdesc[i]->neuron_count, this->layers[i - 1]->get_output_len());
                break;
            case CONVOLUTIONAL:
                ///Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int stride=1)
                this->layers[i] = new Convolutional(this->layers[i - 1]->get_output_row(), this->layers[i - 1]->get_output_col(),
                                                    this->layers[i - 1]->get_mapcount(), layerdesc[i]->row, layerdesc[i]->col,
                                                    layerdesc[i]->mapcount, SIGMOID, layerdesc[i + 1]->layer_type, p, layerdesc[i]->vertical_stride, layerdesc[i]->horizontal_stride);
                break;
            case MAX_POOLING:
                this->layers[i] = new Pooling(layerdesc[i]->row, layerdesc[i]->col, MAX_POOLING, this->layers[i - 1]->get_mapcount(), this->layers[i - 1]->get_output_row(),
                                              this->layers[i - 1]->get_output_col(), layerdesc[i + 1]->layer_type);
                break;
            default:
                cerr << "Unknown layer type: " << layerdesc[i]->layer_type << "\n";
                throw std::exception();
        }
    }
}

void Network::store(char *filename)
{
    ///(int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1)
    ///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout
    ofstream network_params (filename, ios::out | ios::binary);
    if(network_params.is_open())
        {
            network_params.write(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->input_col )), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->costfunction_type)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->dropout_probability)), sizeof(double));
            for(int i=0; i<this->layers_num; i++)
                {
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->layer_type)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->neuron_type)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->neuron_count)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->col)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->mapcount)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->vertical_stride)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->horizontal_stride)), sizeof(int));
                }
            for(int i = -1; i < this->layers_num; i++)
                {
                    this->layers[i]->store(network_params);
                }
            network_params.close();
        }
    else
        {
            cerr << "Unable to open the file:" << '"' << filename << '"' << endl;
            throw exception();
        }
}

inline void Network::feedforward(Matrix **input)
{
    this->layers[-1]->set_input(input);
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers[i]->layers_output(this->layers[i - 1]->get_output());
        }
}

double Network::cost(Matrix &required_output, int req_outp_indx, int test_data_len)
{
    double helper = 0, result = 0;
    switch(this->costfunction_type)
        {
        case QUADRATIC_CF:
            /// 1/2 * ||y(x) - a||^2
            for(int i = 0; i < this->layers[this->layers_num - 1]->get_output_len(); i++)
                {
                    helper = required_output.data[i][0] - this->layers[this->layers_num - 1]->get_output()[0]->data[i][0];
                    result += helper * helper;
                }
            return (0.5 * result) / (test_data_len * 2.0);
        case CROSS_ENTROPY_CF:
            ///y(x)ln a + (1 - y(x))ln(1 - a)
            for(int i = 0; i < this->layers[this->layers_num - 1]->get_output_len(); i++)
                {
                    helper += required_output.data[i][0] * log(this->layers[this->layers_num - 1]->get_output()[0]->data[i][0]) + (1 - required_output.data[i][0]) *
                                    log(1 - this->layers[this->layers_num - 1]->get_output()[0]->data[i][0]);
                }
            return helper / test_data_len;
        case LOG_LIKELIHOOD_CF:
            result = -1 * log(this->layers[this->layers_num - 1]->get_output()[0]->data[req_outp_indx][0]);
            return result / test_data_len;
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        }
}

Matrix Network::get_output(Matrix **input)
{
    ///TODO modify this function to work with multiple input features...
    this->feedforward(input);
    Matrix ret = *(this->layers[this->layers_num - 1]->get_output()[0]);
    return ret;
}

inline void Network::backpropagate(MNIST_data *trainig_data, Layers_features **nabla)
{
    ///currently the final layer has to be a clasification layer
    this->feedforward(trainig_data->input);
    Matrix **delta = new Matrix* [1];
    delta = this->layers[layers_num - 1]->get_output_error(this->layers[layers_num - 2]->get_output(),
                                                    trainig_data->required_output, this->costfunction_type);
    nabla[this->layers_num - 1]->fmap[0]->biases[0][0] = delta[0][0];
    nabla[this->layers_num - 1]->fmap[0]->weights[0][0] = delta[0][0] * this->layers[this->layers_num - 2]->get_output()[0]->transpose();
    /*passing backwards the error*/
    for(int i = this->layers_num - 2; i >= 0; i--)
    {
        delta = this->layers[i]->backpropagate(this->layers[i - 1]->get_output(), this->layers[i + 1], nabla[i][0].fmap, delta);
    }
}

void Network::stochastic(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, double learning_rate, double regularization_rate)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->backpropagate(minibatches[training_data_index], deltanabla);
        for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
    {
        this->layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, nabla[layer_index]);
        nabla[layer_index]->zero();
    }
}

void Network::momentum_based(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->backpropagate(minibatches[training_data_index], deltanabla);
        for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
    {
        nabla_momentum[layer_index][0] = (nabla_momentum[layer_index][0] * momentum) + (nabla[layer_index][0]*(1 - momentum));
        this->layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, nabla_momentum[layer_index]);
        nabla[layer_index]->zero();
    }
}

void Network::nesterov(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, Layers_features **nabla_momentum, double learning_rate, double regularization_rate, double momentum)
{
    for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
    {
        this->layers[layer_index]->update_weights_and_biasses(momentum, regularization_rate, nabla_momentum[layer_index]);
    }
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->backpropagate(minibatches[training_data_index], deltanabla);
        for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
    {
        this->layers[layer_index]->update_weights_and_biasses(-1*momentum, regularization_rate, nabla_momentum[layer_index]);
        nabla_momentum[layer_index][0] = (nabla_momentum[layer_index][0] * momentum) + (nabla[layer_index][0] * learning_rate);
        this->layers[layer_index]->update_weights_and_biasses(1, regularization_rate, nabla_momentum[layer_index]);
        nabla[layer_index]->zero();
    }
}

void Network::rmsprop(MNIST_data **minibatches, int minibatch_len, Layers_features **nabla, Layers_features **deltanabla, Layers_features **squared_grad_moving_avarange, Layers_features **layer_helper, double learning_rate, double regularization_rate, double momentum, double denominator)
{
    for(int training_data_index = 0; training_data_index < minibatch_len; training_data_index++)
    {
        this->backpropagate(minibatches[training_data_index], deltanabla);
        for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
        {
            *nabla[layer_index] += *deltanabla[layer_index];
        }
    }
    for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
    {
        squared_grad_moving_avarange[layer_index][0] = (squared_grad_moving_avarange[layer_index][0] * momentum) + (nabla[layer_index][0].square_element_by() * (1 - momentum));
        layer_helper[layer_index][0] = nabla[layer_index][0] / (squared_grad_moving_avarange[layer_index][0].sqroot() + denominator);
        this->layers[layer_index]->update_weights_and_biasses(learning_rate, regularization_rate, layer_helper[layer_index]);
        nabla[layer_index]->zero();
    }
}

void Network::gradient_descent_variant(int variant, MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost, double regularization_rate,
                                      double denominator, double momentum, MNIST_data **test_data, int minibatch_count , int test_data_len,  int trainingdata_len)
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
    int randomnum;
    int biascnt;
    double previoius_learning_cost = 0;
    double lr, reg;
    Matrix helper(this->layers[this->layers_num - 1]->get_output_row(), 1);
    Matrix dropout_neurons[this->layers_num - 1];
    chrono::time_point<chrono::system_clock> start, end_training;
    chrono::duration<double> epoch_duration, overall_duration;
    Layers_features **nabla, **deltanabla, **helper_1, **helper_2;
    try
    {
        nabla = new Layers_features* [this->layers_num];
        deltanabla = new Layers_features* [this->layers_num];
        helper_1 = new Layers_features* [this->layers_num];
        helper_2 = new Layers_features* [this->layers_num];
        for(int i = 0; i < this->layers_num; i++)
        {
            ///Layers_features(int mapcount, int row, int col, int depth, int biascnt);
            if((this->layers[i]->get_layer_type() == FULLY_CONNECTED) or (this->layers[i]->get_layer_type() == SOFTMAX))
                biascnt = this->layers[i]->get_weights_row();
            else
                biascnt = 1;
            nabla[i] = new Layers_features(this->layers[i]->get_mapcount(),
                                           this->layers[i]->get_weights_row(),
                                           this->layers[i]->get_weights_col(),
                                           this->layers[i]->get_mapdepth(),
                                           biascnt);
            nabla[i]->zero();
            deltanabla[i] = new Layers_features(this->layers[i]->get_mapcount(),
                                                this->layers[i]->get_weights_row(),
                                                this->layers[i]->get_weights_col(),
                                                this->layers[i]->get_mapdepth(),
                                                biascnt);
            helper_1[i] = new Layers_features(this->layers[i]->get_mapcount(),
                                                this->layers[i]->get_weights_row(),
                                                this->layers[i]->get_weights_col(),
                                                this->layers[i]->get_mapdepth(),
                                                biascnt);
            helper_1[i]->zero();
            helper_2[i] = new Layers_features(this->layers[i]->get_mapcount(),
                                                this->layers[i]->get_weights_row(),
                                                this->layers[i]->get_weights_col(),
                                                this->layers[i]->get_mapdepth(),
                                                biascnt);
        }
    }
    catch(bad_alloc& ba)
    {
        cerr<<"operator new failed in the function: Network::update_weights_and_biasses"<<endl;
        return;
    }
    Layers_features** nabla_momentum = helper_1;
    Layers_features** squared_grad_moving_avarange = helper_1;
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
                while(this->layers[dropout_index]->get_layer_type() != FULLY_CONNECTED and this->layers[dropout_index]->get_layer_type() != SOFTMAX)
                {
                    dropout_index++;
                }
                dropout_neurons[dropout_index] = this->layers[dropout_index]->drop_out_some_neurons(dropout_probability, NULL);
                dropout_index++;
                for(dropout_index; dropout_index < this->layers_num - 1; dropout_index++)
                {
                    dropout_neurons[dropout_index] = this->layers[dropout_index]->drop_out_some_neurons(dropout_probability, &dropout_neurons[dropout_index-1]);
                }
                this->layers[layers_num - 1]->drop_out_some_neurons(0.0, &dropout_neurons[layers_num - 2]);
                for(int layerindex = 0; layerindex < this->layers_num; layerindex++)
                {
                    if(this->layers[layerindex]->get_layer_type() == FULLY_CONNECTED or this->layers[layerindex]->get_layer_type() == SOFTMAX)
                    {
                        delete nabla[layerindex];
                        delete deltanabla[layerindex];
                        biascnt = this->layers[layerindex]->get_weights_row();
                        nabla[layerindex] = new Layers_features(this->layers[layerindex]->get_mapcount(),
                                                       this->layers[layerindex]->get_weights_row(),
                                                       this->layers[layerindex]->get_weights_col(),
                                                       this->layers[layerindex]->get_mapdepth(),
                                                       biascnt);
                        deltanabla[layerindex] = new Layers_features(this->layers[layerindex]->get_mapcount(),
                                                            this->layers[layerindex]->get_weights_row(),
                                                            this->layers[layerindex]->get_weights_col(),
                                                            this->layers[layerindex]->get_mapdepth(),
                                                            biascnt);
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
                    this->rmsprop(minibatches[j], minibatch_len, nabla, deltanabla, helper_1, helper_2, lr, reg, momentum, denominator);
                    break;
                default:
                    throw invalid_argument("Unknown gradient descent variant!");
            }
            if(this->dropout_probability != 0)
            {
                int dropout_index = 0;
                while(this->layers[dropout_index]->get_layer_type() != FULLY_CONNECTED and this->layers[dropout_index]->get_layer_type() != SOFTMAX)
                {
                    dropout_index++;
                }
                this->layers[dropout_index]->restore_neurons(NULL);
                dropout_index++;
                for(dropout_index; dropout_index < this->layers_num - 1; dropout_index++)
                {
                    this->layers[dropout_index]->restore_neurons(&dropout_neurons[dropout_index-1]);
                }
                this->layers[layers_num - 1]->restore_neurons(&dropout_neurons[layers_num - 2]);
            }
        }
        for(int layer_index = 0; layer_index < this->layers_num; layer_index++)
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
    for(int i = 0; i < this->layers_num; i++)
    {
        delete nabla[i];
        delete deltanabla[i];
        delete helper_1[i];
        delete helper_2[i];
    }
    delete[] nabla;
    delete[] deltanabla;
    delete[] helper_1;
    delete[] helper_2;
}

void Network::stochastic_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    this->gradient_descent_variant(STOCHASTIC, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 0, 0, test_data, minibatch_count, test_data_len, trainingdata_len);
}

void Network::momentum_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    this->gradient_descent_variant(MOMENTUM, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 0, momentum, test_data, minibatch_count, test_data_len, trainingdata_len);
}

void Network::nesterov_accelerated_gradient(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    this->gradient_descent_variant(NESTEROV, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 0, momentum, test_data, minibatch_count, test_data_len, trainingdata_len);
}

void Network::rmsprop(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost,
                                            double regularization_rate, double denominator, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    this->gradient_descent_variant(RMSPROP, training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate,
                                 denominator, momentum, test_data, minibatch_count, test_data_len, trainingdata_len);
}

Accuracy Network::check_accuracy(MNIST_data **test_data, int test_data_len, int epoch, bool monitor_learning_cost, double regularization_rate)
{
    int learning_accuracy;
    double learning_cost, squared_sum = 0, avarange_confidence = 0;
    Matrix helper(this->layers[this->layers_num - 1]->get_output_row(), 1);
    Matrix output;
    int mapcount, mapdepth;
    Feature_map **fmaps;
    chrono::time_point<chrono::system_clock> start, end_testing;
    chrono::duration<double> test_duration;
    for(int i = 0; i < this->layers[this->layers_num - 1]->get_output_row(); i++)
    {
        helper.data[i][0] = 0;
    }
    learning_accuracy = learning_cost = 0;
    start = chrono::system_clock::now();
    for(int j = 0; j < test_data_len; j++)
    {
        ///TODO this is an errorprone as well
        output = this->get_output(test_data[j]->input);
        if(getmax(output.data) == test_data[j]->required_output.data[0][0])
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
        for(int layerindex = 0; layerindex < this->layers_num; layerindex++)
        {
            if(this->layers[layerindex]->get_layer_type() != POOLING)
            {
                fmaps = this->layers[layerindex]->get_feature_maps();
                mapcount = this->layers[layerindex]->get_mapcount();
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
