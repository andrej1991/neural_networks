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

using namespace std;


inline int get_neuron_type(YAML::const_iterator &it, int i)
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


int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        cerr << "You must provide the yaml config of the neural network!\n" << endl;
        throw exception();
    }
    cout << "Loading the " << argv[1] << " file.\n";
    YAML::Node config = YAML::LoadFile(argv[1]);
    string training_input = config["training_input"].as<string>();
    string required_training_output = config["required_training_output"].as<string>();
    string validation_input = config["validation_input"].as<string>();
    string required_validation_output = config["required_validation_output"].as<string>();
    int input_row = config["input_row"].as<int>();
    int input_col = config["input_col"].as<int>();
    int input_channel_count = config["input_channel_count"].as<int>();
    int output_size = config["output_size"].as<int>();
    int traninig_data_len = config["traninig_data_len"].as<int>();
    int validation_data_len = config["validation_data_len"].as<int>();
    int layer_count = config["layers"].size();
    string cf = config["cost_function_type"].as<string>();
    int costfunction_type = -1;
    int epochs = config["epochs"].as<int>();
    int cpulimit = 0;
    if(config["cpulimit"])
    {
        cpulimit = config["cpulimit"].as<int>();
    }
    int minibatch_len = config["minibatch_len"].as<int>();
    double learning_rate = config["learning_rate"].as<double>();
    double regularization_rate = config["regularization_rate"].as<double>();
    double dropout_probability = config["dropout_probability"].as<double>();
    if(dropout_probability < 0 or dropout_probability > 1)
    {
        cerr << "the probability of dropout must be between 0 - 1!" << endl;
        throw exception();
    }
    int thread_count;
    if(config["thread_count"])
    {
        thread_count = config["thread_count"].as<int>();
    }
    else
    {
        thread_count = 1;
    }
    int minibatch_count = config["minibatch_count"].as<int>();
    double momentum = config["momentum"].as<double>();
    double denominator = config["denominator"].as<double>();
    int vertical_stride, horizontal_stride;
    if(cf.compare("log_likelihood") == 0)
    {
        costfunction_type = LOG_LIKELIHOOD_CF;
    }else if(cf.compare("quadratic") == 0)
    {
        costfunction_type = QUADRATIC_CF;
    }else if(cf.compare("cross_entropy") == 0)
    {
        costfunction_type = CROSS_ENTROPY_CF;
    }
    else
    {
        cerr << "Unknown cost function found in you configfile!\n";
        throw exception();
    }
    string lt;
    int neuron_type;
    LayerDescriptor *layers[layer_count];
    for(int i=0; i<layer_count; i++)
    {
        for(YAML::const_iterator it=config["layers"][i].begin();it!=config["layers"][i].end();++it)
        {
            lt = it->second["layer_type"].as<string>();
            if(lt.compare("convolutional") == 0)
            {
                neuron_type = get_neuron_type(it, i);
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
                layers[i] = new LayerDescriptor(CONVOLUTIONAL, neuron_type, it->second["weights_row"].as<int>(),
                                                it->second["weights_col"].as<int>(), it->second["feature_map_count"].as<int>(),
                                                vertical_stride, horizontal_stride);
            }
            else if(lt.compare("fully_connected") == 0)
            {
                neuron_type = get_neuron_type(it, i);
                layers[i] = new LayerDescriptor(FULLY_CONNECTED, neuron_type, it->second["weights_row"].as<int>());
            }
            else if(lt.compare("softmax") == 0)
            {
                layers[i] = new LayerDescriptor(SOFTMAX, SIGMOID, it->second["weights_row"].as<int>());
            }
            else if(lt.compare("maxpooling") == 0)
            {
                layers[i] = new LayerDescriptor(MAX_POOLING, -1, it->second["filter_row"].as<int>(), it->second["filter_col"].as<int>());
            }
            else
            {
                cerr << "Unknown layer type found in your configfile!\n";
                throw exception();
            }
        }

    }
    if(layers[layer_count - 1]->row != output_size)
    {
        cerr << "The output size and the neuron count in your output layer isn't equal!\n";
        throw exception();
    }
    ifstream input, required_output, validation_input_data, validation_output_data;
    input.open(training_input, ios::in|ios::binary);
    required_output.open(required_training_output, ios::in|ios::binary);
    validation_input_data.open(validation_input, ios::in|ios::binary);
    validation_output_data.open(required_validation_output, ios::in|ios::binary);
    Data_Loader *m[traninig_data_len];
    Data_Loader *validation[validation_data_len];
    for(int i = 0; i < traninig_data_len; i++)
    {
        m[i] = new Data_Loader(input_row, input_col, output_size, input_channel_count);
        //m[i]->load_MNIST(input, required_output);
        m[i]->load_CIFAR(input);
    }
    for(int i = 0; i < validation_data_len; i++)
    {
        validation[i] = new Data_Loader(input_row, input_col, 1, input_channel_count);
        //validation[i]->load_MNIST(validation_input_data, validation_output_data);
        validation[i]->load_CIFAR(validation_input_data);
    }

    Network n1(layer_count, layers, input_row, input_col, input_channel_count);
    /*Network n2(layer_count, layers, input_row, input_col, input_channel_count);
    Network n3(layer_count, layers, input_row, input_col, input_channel_count);
    Network n4(layer_count, layers, input_row, input_col, input_channel_count);*/


    //StochasticGradientDescent learning(n1, costfunction_type, dropout_probability);
    StochasticGradientDescentMultiThread learning(n1, costfunction_type, dropout_probability, thread_count);
    learning.monitor_training_duration = true;

    if(cpulimit > 0)
    {
        pid_t pid = getpid();
        string command = "cpulimit -p " + to_string(pid) + " -l " + to_string(cpulimit) + " &";
        cout << command << endl;
        system(command.c_str());
    }
    //cout << "stohastic gradient descent\n";
    //learning.check_accuracy(validation, 10, 0, true);
    //n1.dropout_probability = dropout_probability;
    //learning.stochastic_gradient_descent(m, epochs, minibatch_len, learning_rate, true, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);
    //learning.momentum_gradient_descent(m, epochs, minibatch_len, learning_rate, momentum, true, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);

    //learning.nesterov_accelerated_gradient(m, epochs, minibatch_len, learning_rate, momentum, true, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);
    //cout << "RMSprop\n";


    learning.rmsprop(m, epochs, minibatch_len, learning_rate, momentum, true, regularization_rate, denominator, validation, minibatch_count, validation_data_len, traninig_data_len);


    input.close();
    required_output.close();
    validation_input_data.close();
    validation_output_data.close();
    return 0;
}
