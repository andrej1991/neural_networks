#include "network.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "random.h"
#include "additional.h"
#include <math.h>
#include "layers/layers.h"

using namespace std;

Network::Network(int layers_num, LayerDescriptor **layerdesc, int inputpixel_count, int costfunction_type,  bool dropout):
                dropout(dropout), inputpixel_count(inputpixel_count)
{
    try
    {
        this->costfunction_type = costfunction_type;
        this->total_layers_num = layers_num + 1;
        this->layers_num = layers_num;
        this->layers = new Layer* [this->total_layers_num];
        Padding p;
        if(layerdesc[0]->layer_type == FULLY_CONNECTED)
            this->layers[0] = new InputLayer(inputpixel_count, 1, 1, SIGMOID, p, FULLY_CONNECTED);
        else
            this->layers[0] = new InputLayer(28, 28, 1, SIGMOID, p, CONVOLUTIONAL);
        this->layers += 1;
        for(int i = 0; i < layers_num; i++)
            {
                switch(layerdesc[i]->layer_type)
                {
                case FULLY_CONNECTED:
                    this->layers[i] = new FullyConnected(layerdesc[i]->neuron_count, this->layers[i - 1]->get_outputlen(), layerdesc[i]->neuron_type);
                    break;
                case CONVOLUTIONAL:
                    ///Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int stride=1)
                    this->layers[i] = new Convolutional(28, 28, 1, layerdesc[i]->row, layerdesc[i]->col, layerdesc[i]->mapcount, SIGMOID, layerdesc[i+1]->layer_type, p, 1);
                    break;
                default:
                    cerr << "Unknown layer type: " << layerdesc[i]->layer_type << "\n";
                    throw std::exception();
                }
            }
    }
    catch(bad_alloc &ba)
        {
            cerr << "bad alloc in Network::constructor" << endl;
        }
}

Network::~Network()
{
    this->layers -= 1;
    for(int i = 0; i <= layers_num; i++)
        {
            delete this->layers[i];
        }
    delete this->layers;
}

inline void Network::feedforward(Matrice **input)
{
    this->layers[-1]->set_input(input);
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers[i]->layers_output(this->layers[i - 1]->get_output());
        }
}

double Network::cost(Matrice &required_output)
{
    double helper = 0, result = 0;
    switch(this->costfunction_type)
        {
        case QUADRATIC_CF:
            /// 1/2 * ||y(x) - a||^2
            for(int i = 0; i < this->layers[this->layers_num - 1]->get_outputlen(); i++)
                {
                    helper = required_output.data[i][0] - this->layers[this->layers_num - 1]->get_output()[0]->data[i][0];
                    result += helper * helper;
                }
            return (1/2) * result;
        case CROSS_ENTROPY_CF:
            ///y(x)ln a + (1 - y(x))ln(1 - a)
            for(int i = 0; i < this->layers[this->layers_num - 1]->get_outputlen(); i++)
                {
                    helper += required_output.data[i][0] * log(this->layers[this->layers_num - 1]->get_output()[0]->data[i][0]) + (1 - required_output.data[i][0]) *
                                    log(1 - this->layers[this->layers_num - 1]->get_output()[0]->data[i][0]);
                }
            return helper;
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
            this->layers[i]->backpropagate(this->layers[i - 1]->get_output(),
                                           this->layers[i + 1]->get_feature_maps(), nabla[i][0].fmap, delta,
                                           nabla[i+1]->get_fmap_count());
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
                    if(this->layers[i]->get_layer_type() == FULLY_CONNECTED)
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
                    //*w[j] += *dnw[j];
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

void Network::load(char *filename)
{
    /*ifstream inp;
    inp.open(filename, ios::in);
    int f_layer, f_row, f_col;
    double data;
    Matrice *w, *b;
    inp.read((char*)&f_layer, 4);
    for(int i = 0; i < f_layer; i++)
        {
            inp.read((char*)&f_row, 4);
            inp.read((char*)&f_col, 4);
            w = new Matrice(f_row, f_col);
            b = new Matrice(f_row, 1);
            for(int j = 0; j < f_row; j++)
                {
                    for(int k = 0; k < f_col; k++)
                        {
                            inp.read((char*)&data, sizeof(double));
                            w->data[j][k] = data;
                        }
                }
            this->layers[i]->set_weights(w);
            for(int l = 0; l < f_row; l++)
                {
                    inp.read((char*)&data, sizeof(double));
                    b->data[l][0] = data;
                }
            this->layers[i]->set_biases(b);
            delete w;
            delete b;
        }
    inp.close();*/
}

void Network::store(char *filename)
{
    ;
}

void Network::stochastic_gradient_descent(MNIST_data **training_data, int epochs, int epoch_len, double learning_rate, bool monitor_learning_cost,
                                            double regularization_rate, MNIST_data **test_data, int minibatch_count, int test_data_len, int trainingdata_len)
{
    if(minibatch_count < 0)
        {
            minibatch_count = trainingdata_len / epoch_len;
        }
    MNIST_data *minibatches[minibatch_count][epoch_len];
    ifstream rand;
    rand.open("/dev/urandom", ios::in);
    int break_counter = 0;
    int learning_accuracy, learnig_cost_counter = 0;
    double learning_cost, previoius_learning_cost = 0;
    //double **helper = new double* [this->layers[this->layers_num - 1]->get_outputlen()];
    Matrice helper(this->layers[this->layers_num - 1]->get_outputlen(), 1);
    for(int i = 0; i < this->layers[this->layers_num - 1]->get_outputlen(); i++)
        {
            helper.data[i][0] == 0;
        }
    Matrice output;
    for(int i = 0; i < epochs; i++)
        {
            for(int j = 0; j < minibatch_count; j++)
                {
                    for(int k = 0; k < epoch_len; k++)
                        {
                            minibatches[j][k] = training_data[random(0, trainingdata_len, rand)];
                        }
                }

            for(int j = 0; j < minibatch_count; j++)
                {
                    this->update_weights_and_biasses(minibatches[j], epoch_len, trainingdata_len, learning_rate, regularization_rate);
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
                                    if(j > 0)
                                        helper.data[(int)test_data[j - 1]->required_output.data[0][0]][0] = 0;
                                    helper.data[(int)test_data[j]->required_output.data[0][0]][0] = 1;
                                    learning_cost += this->cost(helper);
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

void Network::test(MNIST_data **d, MNIST_data **v)
{
    this->stochastic_gradient_descent(d, 30, 10, 3, true, 10, v, 500);
    //this->get_output(v[0]->input);
}
