#include <iostream>
#include <fstream>
#include <yaml.h>
#include <string>
#include "MNIST_data.h"
#include "network.h"
#include "matrice.h"
#include "random.h"
#include <iosfwd>

using namespace std;


inline int get_neuron_type(YAML::Node &config, int i, string &layerstr)
{
    string nt = config["layers"][i][layerstr]["neuron_type"].as<string>();
    if(nt.compare("sigmoid"))
    {
        return SIGMOID;
    }else if(nt.compare("relu"))
    {
        return RELU;
    }else if(nt.compare("leaky_relu"))
    {
        return LEAKY_RELU;
    }
    else
    {
        cerr << "Unknown neuron type found in your config file\n";
        throw exception();
    }
}


int main()
{
    YAML::Node config = YAML::LoadFile("../data/config.yaml");
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
    int layer_count = config["layer_count"].as<int>();
    string cf = config["cost_function_type"].as<string>();
    int costfunction_type = -1;
    if(cf.compare("log_likelihood"))
    {
        costfunction_type = LOG_LIKELIHOOD_CF;
    }else if(cf.compare("quadratic"))
    {
        costfunction_type = QUADRATIC_CF;
    }else if(cf.compare("cross_entropy"))
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
        string layerstr = string("layer-") + to_string(i);
        lt = config["layers"][i][layerstr]["layer_type"].as<string>();
        if(lt.compare("convolutional") == 0)
        {
            neuron_type = get_neuron_type(config, i, layerstr);
            layers[i] = new LayerDescriptor(CONVOLUTIONAL, neuron_type, config["layers"][i][layerstr]["weights_row"].as<int>(),
                                            config["layers"][i][layerstr]["weights_col"].as<int>(),
                                            config["layers"][i][layerstr]["feature_map_count"].as<int>());
        }else if(lt.compare("fully_connected") == 0)
        {
            neuron_type = get_neuron_type(config, i, layerstr);
            layers[i] = new LayerDescriptor(FULLY_CONNECTED, neuron_type, config["layers"][i][layerstr]["weights_row"].as<int>());
        }else if(lt.compare("softmax") == 0)
        {
            layers[i] = new LayerDescriptor(SOFTMAX, SIGMOID, config["layers"][i][layerstr]["weights_row"].as<int>());
        }
        else
        {
            cerr << "Unknown layer type found in your configfile!\n";
            throw exception();
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
    Network n(layer_count, layers, input_row, input_col, input_channel_count, costfunction_type);

    n.test(m, validation);


    input.close();
    required_output.close();
    validation_input_data.close();
    validation_output_data.close();
    return 0;
}
