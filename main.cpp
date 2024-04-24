#include <iostream>
#include <fstream>
#include <yaml.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>

#include "data_loader/data_loader.h"
#include "network.h"
#include "matrix/matrix.h"
#include <random>
#include <iosfwd>
#include "SGD.h"


#include <thread>
#include <chrono>

#include "additional.h"

using namespace std;


int get_neuron_type(YAML::const_iterator &it)
{
    string nt = it->second["neuron_type"].as<string>();
    if(nt.compare("sigmoid") == 0)
    {
        return SIGMOID;
    }else if(nt.compare("relu") == 0)
    {
        return RELU;
    }else if(nt.compare("leaky_relu") == 0)
    {
        return LEAKY_RELU;
    }
    else if(nt.compare("tanh") == 0)
    {
        return TANH;
    }
    else
    {
        cerr << "Unknown neuron type found in your config file\n";
        throw exception();
    }
}

void get_strides(YAML::const_iterator &it, int &vertical_stride, int &horizontal_stride)
{
    try
    {
        vertical_stride = it->second["vertical_stride"].as<int>();
    }
    catch(YAML::InvalidNode)
    {
        vertical_stride = 1;
    }
    try
    {
        horizontal_stride = it->second["horizontal_stride"].as<int>();
    }
    catch(YAML::InvalidNode)
    {
        horizontal_stride = 1;
    }
}

int get_cpu_limit(YAML::Node &config)
{
    if(config["cpulimit"])
    {
        return config["cpulimit"].as<int>();
    }
    return 0;
}

double get_dropout(YAML::Node &config)
{
    double dropout_probability = config["dropout_probability"].as<double>();
    if(dropout_probability < 0 or dropout_probability > 1)
    {
        cerr << "the probability of dropout must be between 0 - 1!" << endl;
        throw exception();
    }
    return dropout_probability;
}

int get_threadcount(YAML::Node &config)
{
    if(config["thread_count"])
    {
        return config["thread_count"].as<int>();
    }
    return 1;
}

int get_costfunction(YAML::Node &config)
{
    string cf = config["cost_function_type"].as<string>();
    if(cf.compare("log_likelihood") == 0)
    {
        return LOG_LIKELIHOOD_CF;
    }else if(cf.compare("quadratic") == 0)
    {
        return QUADRATIC_CF;
    }else if(cf.compare("cross_entropy") == 0)
    {
        return CROSS_ENTROPY_CF;
    }
    else
    {
        cerr << "Unknown cost function found in you configfile!\n";
        throw exception();
    }
}

int get_layercount(YAML::Node &config)
{
    return config["layers"].size();
}

///TODO check if all the connections are unique
vector<string> get_connections(YAML::const_iterator config)
{
    vector<string> connections;
    if(config->second["connects_to"])
    {
        for(int i=0; i < config->second["connects_to"].size(); i++)
        {
            connections.push_back(config->second["connects_to"][i].as<string>());
        }
    }
    /*try
    {
        for(int i=0; i < config->second["connects_to"].size(); i++)
        {
            connections.push_back(config->second["connects_to"][i].as<string>());
        }
    }
    catch(YAML::InvalidNode)
    {
        connections.clear();
    }*/
    return connections;
}

bool check_if_recurrent(LayerDescriptor **layers, vector<string> &connections, int index)
{
    for(string conn : connections)
    {
        for(int j = 0; j <= index; j++)
        {
            if(layers[j]->get_name() == conn)
            {
                cout << "In the layer: " << layers[index]->get_name() << " the connection: " << conn << " connects to the layer " << layers[j]->get_name() << " which makes it recurrent.\n";
                return true;
            }
        }
    }
    return false;
}

int get_layers(LayerDescriptor **layers, YAML::Node &config)
{
    int vertical_stride, horizontal_stride;
    int layer_count = get_layercount(config);
    string lt;
    int neuron_type;
    vector<string> connections;
    for(int i=0; i<layer_count; i++)
    {
        connections.clear();
        for(YAML::const_iterator it=config["layers"][i].begin();it!=config["layers"][i].end();++it)
        {
            lt = it->second["layer_type"].as<string>();
            connections = get_connections(it);
            if(i < layer_count-1)
            {
                ///TODO: check if that connection is already present
                connections.insert(connections.begin(), config["layers"][i+1].begin()->first.as<string>());
            }
            if(lt.compare("convolutional") == 0)
            {
                neuron_type = get_neuron_type(it);
                get_strides(it, vertical_stride, horizontal_stride);
                layers[i] = new LayerDescriptor(CONVOLUTIONAL, neuron_type, it->second["weights_row"].as<int>(), connections, it->first.as<string>(),
                                                it->second["weights_col"].as<int>(), it->second["feature_map_count"].as<int>(),
                                                vertical_stride, horizontal_stride);
            }
            else if(lt.compare("fully_connected") == 0)
            {
                neuron_type = get_neuron_type(it);
                layers[i] = new LayerDescriptor(FULLY_CONNECTED, neuron_type, it->second["weights_row"].as<int>(), connections, it->first.as<string>());
            }
            else if(lt.compare("softmax") == 0)
            {
                layers[i] = new LayerDescriptor(SOFTMAX, SIGMOID, it->second["weights_row"].as<int>(), connections, it->first.as<string>());
            }
            else if(lt.compare("maxpooling") == 0)
            {
                layers[i] = new LayerDescriptor(MAX_POOLING, -1, it->second["filter_row"].as<int>(), connections, it->first.as<string>(), it->second["filter_col"].as<int>());
            }
            else
            {
                cerr << "Unknown layer type found in your configfile!\n";
                throw exception();
            }
            if(check_if_recurrent(layers, connections, i))
            {
                cerr << "Recurrent units are currently not supported!\n";
                throw exception();
            }
        }
    }
    return layer_count;
}

inline int get_input_row(YAML::Node &config)
{
    return config["input_row"].as<int>();
}

inline int get_input_col(YAML::Node &config)
{
    return config["input_col"].as<int>();
}

inline int get_input_channel_count(YAML::Node &config)
{
    return config["input_channel_count"].as<int>();
}

inline int get_traninig_data_len(YAML::Node &config)
{
    return config["traninig_data_len"].as<int>();
}

inline int get_validation_data_len(YAML::Node &config)
{
    return config["validation_data_len"].as<int>();
}

void load_data(YAML::Node &config, int output_size, Data_Loader **m, Data_Loader **validation)
{
    string training_input = config["training_input"].as<string>();
    string required_training_output = config["required_training_output"].as<string>();
    string validation_input = config["validation_input"].as<string>();
    string required_validation_output = config["required_validation_output"].as<string>();
    ifstream input, required_output, validation_input_data, validation_output_data;
    int input_row = get_input_row(config);
    int input_col = get_input_col(config);
    int input_channel_count = get_input_channel_count(config);
    int traninig_data_len = get_traninig_data_len(config);
    int validation_data_len = get_validation_data_len(config);
    input.open(training_input, ios::in|ios::binary);
    required_output.open(required_training_output, ios::in|ios::binary);
    validation_input_data.open(validation_input, ios::in|ios::binary);
    validation_output_data.open(required_validation_output, ios::in|ios::binary);
    for(int i = 0; i < traninig_data_len; i++)
    {
        m[i] = new Data_Loader(input_row, input_col, output_size, input_channel_count);
        m[i]->load_MNIST(input, required_output);
        //m[i]->load_CIFAR(input);
    }
    for(int i = 0; i < validation_data_len; i++)
    {
        validation[i] = new Data_Loader(input_row, input_col, 1, input_channel_count);
        validation[i]->load_MNIST(validation_input_data, validation_output_data);
        //validation[i]->load_CIFAR(validation_input_data);
    }
    input.close();
    required_output.close();
    validation_input_data.close();
    validation_output_data.close();
}

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        cerr << "You must provide the yaml config of the neural network!\n" << endl;
        throw exception();
    }
    cout << "Loading the " << argv[1] << " file.\n";
    YAML::Node config = YAML::LoadFile(argv[1]);
    int input_row = get_input_row(config);
    int input_col = get_input_col(config);
    int input_channel_count = get_input_channel_count(config);
    int traninig_data_len = get_traninig_data_len(config);
    int validation_data_len = get_validation_data_len(config);
    int costfunction_type = get_costfunction(config);
    int epochs = config["epochs"].as<int>();
    int change_learning_cost = 0;
    if(config["change_learning_cost"])
    {
        change_learning_cost = config["change_learning_cost"].as<int>();
    }
    int cpulimit = get_cpu_limit(config);
    int minibatch_len = config["minibatch_len"].as<int>();
    double learning_rate = config["learning_rate"].as<double>();
    double regularization_rate = config["regularization_rate"].as<double>();
    double dropout_probability = get_dropout(config);
    int thread_count = get_threadcount(config);
    int minibatch_count = config["minibatch_count"].as<int>();
    double momentum = config["momentum"].as<double>();
    double denominator = config["denominator"].as<double>();
    LayerDescriptor **layers = new LayerDescriptor* [get_layercount(config)];
    int layer_count = get_layers(layers, config);
    /*for(int i = 0; i < layer_count; i++)
    {
        cout << layers[i]->name << endl;
        for(string s : layers[i]->output_connections) {
            cout << "  " << s << endl;
        }
    }*/
    int output_size = layers[layer_count - 1]->row;
    Data_Loader **m, **validation;
    m = new Data_Loader* [traninig_data_len];
    validation = new Data_Loader* [validation_data_len];
    load_data(config, output_size, m, validation);
    Network n1(layer_count, layers, input_row, input_col, input_channel_count);
    /*Network n2(layer_count, layers, input_row, input_col, input_channel_count);
    Network n3(layer_count, layers, input_row, input_col, input_channel_count);
    Network n4(layer_count, layers, input_row, input_col, input_channel_count);*/


    StochasticGradientDescent learning(n1, costfunction_type, dropout_probability);
    //StochasticGradientDescentMultiThread learning(n1, costfunction_type, dropout_probability, thread_count);
    learning.monitor_training_duration = true;

    if(cpulimit > 0)
    {
        pid_t pid = getpid();
        string command = "cpulimit -p " + to_string(pid) + " -l " + to_string(cpulimit) + " &";
        cout << command << endl;
        system(command.c_str());
    }
    n1.get_output(m[0]->input);
    //cout << "stohastic gradient descent\n";
    //learning.check_accuracy(validation, 10, 0, true);
    //n1.dropout_probability = dropout_probability;
    learning.stochastic_gradient_descent(m, epochs, minibatch_len, learning_rate, change_learning_cost, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);
    //learning.momentum_gradient_descent(m, epochs, minibatch_len, learning_rate, momentum, change_learning_cost, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);

    //learning.nesterov_accelerated_gradient(m, epochs, minibatch_len, learning_rate, momentum, change_learning_cost, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);
    //cout << "RMSprop\n";
    //Accuracy A = learning.check_accuracy(validation, 100, 0, 1, regularization_rate);
    //cout << "cost: " << A.total_cost << " correct answers: " << A.correct_answers << endl;

    //learning.rmsprop(m, epochs, minibatch_len, learning_rate, momentum, change_learning_cost, regularization_rate, denominator, validation, minibatch_count, validation_data_len, traninig_data_len);
    //A = learning.check_accuracy(m, 50000, 0, 1, regularization_rate);
    //cout << "cost: " << A.total_cost << " correct answers: " << A.correct_answers << endl;

    return 0;
}
