#include <iostream>
#include <fstream>
#include <yaml.h>
#include <string>
#include "MNIST_data.h"
#include "network.h"
#include "matrix.h"
#include <random>
#include <iosfwd>


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
    int minibatch_len = config["minibatch_len"].as<int>();
    double learning_rate = config["learning_rate"].as<double>();
    double regularization_rate = config["regularization_rate"].as<double>();
    double dropout_probability = config["dropout_probability"].as<double>();
    if(dropout_probability < 0 or dropout_probability > 1)
    {
        cerr << "the probability of dropout must be between 0 - 1!" << endl;
        throw exception();
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
    MNIST_data *m[traninig_data_len];
    MNIST_data *validation[validation_data_len];
    for(int i = 0; i < traninig_data_len; i++)
    {
        m[i] = new MNIST_data(input_row, input_col, output_size, 1);
        m[i]->load_data(input, required_output);
    }
    for(int i = 0; i < validation_data_len; i++)
    {
        validation[i] = new MNIST_data(input_row, input_col, 1, 1);
        validation[i]->load_data(validation_input_data, validation_output_data);
    }
    Network n1(layer_count, layers, input_row, input_col, input_channel_count, costfunction_type);
    //Network n2(layer_count, layers, input_row, input_col, input_channel_count, costfunction_type);
    //Network n3(layer_count, layers, input_row, input_col, input_channel_count, costfunction_type);
    Network n4(layer_count, layers, input_row, input_col, input_channel_count, costfunction_type);


    /*Pooling p(2, 2, MAX_POOLING, 1, 11, 11);
    Matrix *inp = new Matrix(11, 11);

    for(int i = 0; i<11;i++)
        for(int j=0;j<11;j++)
        {
            inp->data[i][j] = i+j;
        }
    p.layers_output(&inp);
    print_mtx(*inp);
    print_mtx((p.get_output())[0][0]);*/


    //Network n1("../data/fully_conn.bin");
    //Network n2("../data/fully_conn.bin");
    //Network n3("../data/fully_conn.bin");
    //Network n4("../data/fully_conn.bin");
    /*Network n1("../data/conv.bin");
    Network n2("../data/conv.bin");
    Network n3("../data/conv.bin");
    Network n4("../data/conv.bin");*/

    /*cout << "stohastic gradient descent\n";
    //n1.check_accuracy(validation, 10, 0, true);
    n1.dropout_probability = dropout_probability;
    n1.stochastic_gradient_descent(m, epochs, minibatch_len, learning_rate, true, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);
    /*cout << "momentum based gradient descent\n";
    n2.momentum_gradient_descent(m, epochs, minibatch_len, learning_rate, momentum, true, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);
    cout << "nesterov accelerated gradient\n";
    n3.nesterov_accelerated_gradient(m, epochs, minibatch_len, learning_rate, momentum, true, regularization_rate, validation, minibatch_count, validation_data_len, traninig_data_len);*/
    cout << "RMSprop\n";
    n1.test(RMSPROP, m, epochs, minibatch_len, learning_rate, true, regularization_rate, denominator, momentum, validation, minibatch_count, validation_data_len, traninig_data_len);
std::this_thread::sleep_for (std::chrono::seconds(20));
    n4.rmsprop(m, epochs, minibatch_len, learning_rate, momentum, true, regularization_rate, denominator, validation, minibatch_count, validation_data_len, traninig_data_len);

    input.close();
    required_output.close();
    validation_input_data.close();
    validation_output_data.close();
    return 0;
}
