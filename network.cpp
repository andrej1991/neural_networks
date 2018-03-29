#include "network.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "random.h"
#include "additional.h"
#include <math.h>
#include "layers/layers.h"

using namespace std;
///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int costfunction_type = CROSS_ENTROPY_CF, bool dropout = false
Network::Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout):
                dropout(dropout), input_row(input_row), input_col(input_col), input_channel_count(input_channel_count), layers_num(layers_num)
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

Network::Network(char *data)
{
    ///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout
    ///int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1
    ifstream file (data, ios::in|ios::binary);
    if(file.is_open())
        {
            NetworkVarHelper helper;
            int f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_stride;
            LayerDescriptor *dsc[this->layers_num];
            file.read((char*)&helper, sizeof(NetworkVarHelper));
            this->layers_num = helper.layers_num;
            this->input_row = helper.input_row;
            this->input_col = helper.input_col;
            this->input_channel_count = helper.input_channel_count;
            this->costfunction_type = helper.costfunction_type;
            this->dropout = helper.dropout;
            this->total_layers_num = this->layers_num + 1;
            for(int i = 0; i < helper.layers_num; i++)
                {
                    file.read(reinterpret_cast<char *>(&f_layer_type), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_neuron_type), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_neuron_count), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_col), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_mapcount), sizeof(int));
                    file.read(reinterpret_cast<char *>(&f_stride), sizeof(int));
                    dsc[i] = new LayerDescriptor(f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_stride);
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
            cerr << "Unable to open the file:" << '"' << file << '"' << endl;
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
                                                    layerdesc[i]->col, layerdesc[i]->mapcount, layerdesc[i]->stride);
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
                                                    layerdesc[i]->mapcount, SIGMOID, layerdesc[i + 1]->layer_type, p, 1);
                break;
            default:
                cerr << "Unknown layer type: " << layerdesc[i]->layer_type << "\n";
                throw std::exception();
            }
        }
}

inline void Network::feedforward(Matrice **input)
{
    this->layers[-1]->set_input(input);
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers[i]->layers_output(this->layers[i - 1]->get_output());
        }
}

double Network::cost(Matrice &required_output, int req_outp_indx)
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
            return (1/2) * result;
        case CROSS_ENTROPY_CF:
            ///y(x)ln a + (1 - y(x))ln(1 - a)
            for(int i = 0; i < this->layers[this->layers_num - 1]->get_output_len(); i++)
                {
                    helper += required_output.data[i][0] * log(this->layers[this->layers_num - 1]->get_output()[0]->data[i][0]) + (1 - required_output.data[i][0]) *
                                    log(1 - this->layers[this->layers_num - 1]->get_output()[0]->data[i][0]);
                }
            return helper;
        case LOG_LIKELIHOOD_CF:
            result = -1 * log(this->layers[this->layers_num - 1]->get_output()[0]->data[req_outp_indx][0]);
            return result;
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        }
}


inline void Network::backpropagate(MNIST_data *trainig_data, Layers_features **nabla)
{
    ///TODO modfy this function based on multiple input features...
    ///currently the final layer has to be a clasification layer
    ///TODO make some check if the final layer has only one output vector
    this->feedforward(trainig_data->input);
    Matrice **delta = new Matrice* [1];
    delta[0] = new Matrice;
    delta[0][0] = this->layers[layers_num - 1]->get_output_error(this->layers[layers_num - 2]->get_output(),
                                                    trainig_data->required_output, this->costfunction_type);
    nabla[this->layers_num - 1]->fmap[0]->biases[0][0] = delta[0][0];
    nabla[this->layers_num - 1]->fmap[0]->weights[0][0] = delta[0][0] * this->layers[this->layers_num - 2]->get_output()[0]->transpose();
    /*passing backwards the error*/
    for(int i = this->layers_num - 2; i >= 0; i--)
        {
            Feature_map **test = this->layers[i + 1]->get_feature_maps();
            delta = this->layers[i]->backpropagate(this->layers[i - 1]->get_output(),
                                           this->layers[i + 1]->get_feature_maps(), nabla[i][0].fmap, delta,
                                           nabla[i+1]->get_fmap_count());
        }
    //print_mtx(delta[3][0]);
    if(this->layers[0]->get_mapcount() > 1)
        {
            for(int i = 0; i < this->layers[0]->get_mapcount(); i++)
                delete delta[i];
            delete[] delta;
        }
    else
        {
            delete delta[0];
            delete[] delta;
        }
}

void Network::update_weights_and_biasses(MNIST_data **training_data, int training_data_len, int total_trainingdata_len, double learning_rate, double regularization_rate)
{
    Layers_features **nabla, **deltanabla;
    //Matrice **b_bck, **w_bck;
    int *layer_bck, **ind;
    //this->remove_some_neurons(&w_bck, &b_bck, &layer_bck, &ind);
    try
        {
            nabla = new Layers_features* [this->layers_num];
            deltanabla = new Layers_features* [this->layers_num];
            for(int i = 0; i < this->layers_num; i++)
                {
                    ///Layers_features(int mapcount, int row, int col, int depth, int biascnt);
                    int biascnt;
                    if((this->layers[i]->get_layer_type() == FULLY_CONNECTED) or (this->layers[i]->get_layer_type() == SOFTMAX))
                        biascnt = this->layers[i]->get_weights_row();
                    else
                        biascnt = 1;
                    nabla[i] = new Layers_features(this->layers[i]->get_mapcount(),
                                                   this->layers[i]->get_weights_row(),
                                                   this->layers[i]->get_weights_col(),
                                                   this->layers[i]->get_mapdepth(),
                                                   biascnt);
                    deltanabla[i] = new Layers_features(this->layers[i]->get_mapcount(),
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
    for(int i = 0; i < training_data_len; i++)
        {
            this->backpropagate(training_data[i], deltanabla);
            for(int j = 0; j < this->layers_num; j++)
                {
                    *nabla[j] += *deltanabla[j];
                }
        }
    double lr = learning_rate / training_data_len;
    double reg = (1 - learning_rate * (regularization_rate / total_trainingdata_len));
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers[i]->update_weights_and_biasses(lr, reg, nabla[i]);
        }
    //this->add_back_removed_neurons(w_bck, b_bck, layer_bck, ind);
    for(int i = 0; i < this->layers_num; i++)
        {
            delete nabla[i];
            delete deltanabla[i];
        }
    delete[] nabla;
    delete[] deltanabla;
}

Matrice Network::get_output(Matrice **input)
{
    ///TODO modify this function to work with multiple input features...
    this->feedforward(input);
    Matrice ret = *(this->layers[this->layers_num - 1]->get_output()[0]);
    return ret;
}

inline void Network::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void Network::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

/*void Network::load(char *filename)
{
    ;
}*/

void Network::store(char *filename)
{
    ///(int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1)
    ///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout
    ofstream network_params (filename, ios::out | ios::binary);
    if(network_params.is_open())
        {
            //network_params << this->layers_num << this->input_row << this->input_col << this->input_channel_count << this->costfunction_type << this->dropout;
            network_params.write((char*)(&(this->layers_num)), sizeof(int));
            network_params.write((char*)(&(this->input_row)), sizeof(int));
            network_params.write((char*)(&(this->input_col )), sizeof(int));
            network_params.write((char*)(&(this->input_channel_count)), sizeof(int));
            network_params.write((char*)(&(this->costfunction_type)), sizeof(int));
            network_params.write((char*)(&(this->dropout)), sizeof(int));
            for(int i=0; i<this->layers_num; i++)
                {
                    //network_params << this->layerdsc[i]->layer_type << this->layerdsc[i]->neuron_type << this->layerdsc[i]->neuron_count
                    //               << this->layerdsc[i]->col << this->layerdsc[i]->mapcount << this->layerdsc[i]->stride;
                    network_params.write((char*)(&(this->layerdsc[i]->layer_type)), sizeof(int));
                    network_params.write((char*)(&(this->layerdsc[i]->neuron_type)), sizeof(int));
                    network_params.write((char*)(&(this->layerdsc[i]->neuron_count)), sizeof(int));
                    network_params.write((char*)(&(this->layerdsc[i]->col)), sizeof(int));
                    network_params.write((char*)(&(this->layerdsc[i]->mapcount)), sizeof(int));
                    network_params.write((char*)(&(this->layerdsc[i]->stride)), sizeof(int));
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

void Network::stochastic_gradient_descent(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    if(minibatch_count < 0)
        {
            minibatch_count = trainingdata_len / minibatch_len;
        }
    MNIST_data *minibatches[minibatch_count][minibatch_len];
    ifstream rand;
    rand.open("/dev/urandom", ios::in);
    int break_counter = 0;
    int learning_accuracy, learnig_cost_counter = 0;
    double learning_cost, previoius_learning_cost = 0;
    Matrice helper(this->layers[this->layers_num - 1]->get_output_row(), 1);
    for(int i = 0; i < this->layers[this->layers_num - 1]->get_output_row(); i++)
        {
            helper.data[i][0] == 0;
        }
    Matrice output;
    for(int i = 0; i < epochs; i++)
        {
            for(int j = 0; j < minibatch_count; j++)
                {
                    for(int k = 0; k < minibatch_len; k++)
                        {
                            minibatches[j][k] = training_data[random(0, trainingdata_len, rand)];
                        }
                }

            for(int j = 0; j < minibatch_count; j++)
                {
                    this->update_weights_and_biasses(minibatches[j], minibatch_len, trainingdata_len, learning_rate, regularization_rate);
                }
            if(test_data != NULL)
                {
                    learning_accuracy = learning_cost = 0;
                    for(int j = 0; j < test_data_len; j++)
                        {
                            ///TODO this is an errorprone as well
                            output = this->get_output(test_data[j]->input);
                            if(getmax(output.data) == test_data[j]->required_output.data[0][0])
                                {
                                    learning_accuracy++;
                                }
                            if(monitor_learning_cost)
                                {
                                    //if(j > 0)
                                    //    helper.data[(int)test_data[j - 1]->required_output.data[0][0]][0] = 0;
                                    helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 1;
                                    learning_cost += this->cost(helper, test_data[j]->required_output.data[0][0]);
                                    helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 0;
                                }
                        }
                    cout << "set " << i << ": " << learning_accuracy << " out of: " << test_data_len << endl;
                    if(monitor_learning_cost)
                        {
                            cout << "total cost: " << learning_cost << endl;
                            if(abs(learning_cost) > abs(previoius_learning_cost))
                                learnig_cost_counter++;
                            if(learnig_cost_counter == 10)
                                {
                                    learnig_cost_counter = 0;
                                    learning_rate == 0 ? learning_rate = 1 : learning_rate /= 2;
                                    cout << "changing leatning rate to: " << learning_rate << endl;
                                    if((break_counter++) == 50)
                                        {
                                            cout << "the learning rate is too small\n";
                                            break;
                                        }
                                }
                            previoius_learning_cost = learning_cost;
                        }
                }
        }
    rand.close();
}

void Network::check_accuracy(MNIST_data **test_data)
{
    int break_counter = 0;
    int learning_accuracy, learnig_cost_counter = 0;
    double learning_cost, previoius_learning_cost = 0;
    Matrice helper(this->layers[this->layers_num - 1]->get_output_row(), 1);
    Matrice output;
    int test_data_len = 10000;
    bool monitor_learning_cost = true;
    for(int i = 0; i < this->layers[this->layers_num - 1]->get_output_row(); i++)
        {
            helper.data[i][0] == 0;
        }
    learning_accuracy = learning_cost = 0;
    for(int j = 0; j < test_data_len; j++)
        {
            ///TODO this is an errorprone as well
            output = this->get_output(test_data[j]->input);
            if(getmax(output.data) == test_data[j]->required_output.data[0][0])
                {
                    learning_accuracy++;
                }
            if(monitor_learning_cost)
                {
                    //if(j > 0)
                    //    helper.data[(int)test_data[j - 1]->required_output.data[0][0]][0] = 0;
                    helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 1;
                    learning_cost += this->cost(helper, test_data[j]->required_output.data[0][0]);
                    helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 0;
                }
        }
    cout << learning_accuracy << " out of: " << test_data_len << endl;
    if(monitor_learning_cost)
        {
            cout << "total cost: " << learning_cost << endl;
            if(abs(learning_cost) > abs(previoius_learning_cost))
                learnig_cost_counter++;
            previoius_learning_cost = learning_cost;
        }
}

void Network::test(MNIST_data **d, MNIST_data **v)
{
    ///(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, bool monitor_learning_cost, double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
    this->stochastic_gradient_descent(d, 2, 10, 0.1, true, 10, v, 50);
    this->store("/home/andrej/myfiles/Asztal/net.bin");
    //this->get_output(v[0]->input);
}
