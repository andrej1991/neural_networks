#include "network.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "layers/layers.h"

using namespace std;
///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int costfunction_type = CROSS_ENTROPY_CF, bool dropout = false
Network::Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count):
                input_row(input_row), input_col(input_col), input_channel_count(input_channel_count), layers_num(layers_num)
{
    this->total_layers_num = layers_num + 1;
    this->threadcount = 1;
    try
    {
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
    int dummy1=0;
    double dummy2 = 0.0;
    ifstream file (data, ios::in|ios::binary);
    if(file.is_open())
        {
            int f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_vertical_stride, f_horizontal_stride;
            file.read(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_col )), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(dummy1)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(dummy2)), sizeof(double));
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
    int dummy1;
    double dummy2;
    ofstream network_params (filename, ios::out | ios::binary);
    if(network_params.is_open())
        {
            network_params.write(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->input_col )), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(dummy1)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(dummy2)), sizeof(double));
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

void Network::feedforward(Matrix **input, int threadindex)
{
    this->layers[-1]->set_input(input, threadindex);
    for(int i = 0; i < this->layers_num; i++)
    {
        this->layers[i]->layers_output(this->layers[i - 1]->get_output(threadindex), threadindex);
    }
}

Matrix Network::get_output(Matrix **input, int threadindex)
{
    ///TODO modify this function to work with multiple input features...
    this->feedforward(input, threadindex);
    Matrix ret = *(this->layers[this->layers_num - 1]->get_output(threadindex)[0]);
    return ret;
}

void Network::backpropagate(MNIST_data *trainig_data, Layers_features **nabla, int costfunction_type, int threadindex)
{
    ///currently the final layer has to be a clasification layer
    this->feedforward(trainig_data->input, threadindex);
    Matrix **delta = new Matrix* [1];
    delta = this->layers[layers_num - 1]->get_output_error(this->layers[layers_num - 2]->get_output(threadindex),
                                                    trainig_data->required_output, costfunction_type, threadindex);
    nabla[this->layers_num - 1]->fmap[0]->biases[0][0] = delta[0][0];
    nabla[this->layers_num - 1]->fmap[0]->weights[0][0] = delta[0][0] * this->layers[this->layers_num - 2]->get_output(threadindex)[0]->transpose();
    /*passing backwards the error*/
    for(int i = this->layers_num - 2; i >= 0; i--)
    {
        delta = this->layers[i]->backpropagate(this->layers[i - 1]->get_output(threadindex), this->layers[i + 1], nabla[i][0].fmap, delta, threadindex);
    }
}

int Network::get_threadcount()
{
    return this->threadcount;
}

void Network::set_threadcount(int threadcnt)
{
    for(int i = -1; i < this->layers_num; i++)
    {
        this->layers[i]->set_threadcount(threadcnt);
    }
    this->threadcount = threadcnt;
}

