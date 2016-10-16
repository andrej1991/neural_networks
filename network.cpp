#include "network.h"
#include <fstream>
#include <iostream>

using namespace std;

Network::Network(int layers_num, int *layers)
{
    try
    {
        this->layers_num = layers_num;
        this->layers = new int [layers_num];
        for(int i = 0; i < layers_num; i++)
            this->layers[i] = layers[i];
        this->biases = new Matrice* [layers_num];
        this->outputs = new Matrice* [layers_num];
        this->weights = new Matrice* [layers_num];
        for(int i = 0; i < layers_num; i++)
            {
                this->biases[i] = new Matrice(layers[i], 1);
                this->outputs[i] = new Matrice(layers[i], 1);
                this->weights[i] = new Matrice(layers[i], (i - 1) >= 0 ? layers[i - 1] : 1);
            }
    }
    catch(bad_alloc &ba)
        {
            cerr << "bad alloc in Networks constructor" << endl;
        }
    this->initialize_biases();
    this->initialize_weights();
}

Network::~Network()
{
    for(int i = 0; i < layers_num; i++)
        {
            delete this->biases[i];
            delete this->outputs[i];
            delete this->weights[i];
        }
    delete this->biases;
    delete this->outputs;
    delete this->weights;
}

void Network::initialize_biases()
{
    ifstream random;
    random.open("/dev/urandom", ios::in);
    short int val;
    for(int i = 0; i < this->layers_num; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    if(i == 0)
                        val = 0;
                    else
                        random.read((char*)(&val), 2);
                    this->biases[i]->data[j][0] = val;
                    this->biases[i]->data[j][0] /= 63000;
                }
        }
    random.close();
}

void Network::initialize_weights()
{
    ifstream random;
    random.open("/dev/urandom", ios::in);
    short int val;
    for(int i = 0; i < this->layers_num; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    int len = this->get_inputlen(i);
                    for(int k = 0; k < len; k++)
                        {
                            if(i == 0)
                                val = 1;
                            else
                                random.read((char*)(&val), 2);
                            this->weights[i]->data[j][k] = val;
                            this->weights[i]->data[j][k] /= 63000;
                        }
                }
        }
    random.close();
}

inline void Network::layers_output(double **input, int layer)
{
    int inputlen = this->get_inputlen(layer);
    for(int i = 0; i < this->layers[layer]; i++)
        {
            this->outputs[layer]->data[i][0] = this->neuron.sigmoid(this->weights[layer]->data[i], input, this->biases[layer]->data[i][0], inputlen);
        }
}

inline void Network::feedforward(double **input)
{
    this->layers_output(input, 0);
    for(int i = 1; i < this->layers_num; i++)
        {
            this->layers_output(this->outputs[i - 1]->data, i);
        }
}

inline void Network::backpropagate(MNIST_data *trainig_data, Matrice **nabla_b, Matrice **nabla_w)
{
    Matrice multiplied;
    this->feedforward(trainig_data->input);
    Matrice cf_derivate = this->cost_derivate(this->outputs[this->layers_num - 1]->data, trainig_data->required_output);
    Matrice output_derivate = this->derivate_layers_output(this->layers_num - 1, this->outputs[this->layers_num - 2]->data);
    Matrice delta = hadamart_product(cf_derivate, output_derivate);
    //Matrice delta;
    *(nabla_b[this->layers_num - 1]) = delta;
    *(nabla_w[this->layers_num - 1]) = delta * this->outputs[this->layers_num - 2]->transpose();
    /*passing backwards the error*/
    for(int i = this->layers_num - 2; i > 0; i--)
        {
            output_derivate = this->derivate_layers_output(i, this->outputs[i - 1]->data);
            multiplied = this->weights[i + 1]->transpose() * delta;
            delta = hadamart_product(multiplied, output_derivate);
            *(nabla_b[i]) = delta;
            *(nabla_w[i]) = delta * outputs[i -1]->transpose();
        }
}

inline Matrice Network::cost_derivate(double **output, double **required_output)
{
    Matrice mtx(this->layers[this->layers_num - 1], 1);
    for(int i = 0; i < this->layers[this->layers_num - 1]; i++)
        {
            mtx.data[i][0] = output[i][0] - required_output[i][0];
        }
    return mtx;
}

inline Matrice Network::derivate_layers_output(int layer, double **input)
{
    Matrice mtx(this->layers[layer], 1);
    for(int i = 0; i < this->layers[layer]; i++)
        {
            mtx.data[i][0] = this->neuron.sigmoid_derivate(this->weights[layer]->data[i], input, this->biases[layer]->data[i][0], this->get_inputlen(layer));
        }
    return mtx;
}

inline int Network::get_inputlen(int layer)
{
    if(layer == 0)
        {
            return 1;
        }
    else
        {
            return this->layers[layer - 1];
        }
}

void Network::update_weights_and_biasses(MNIST_data **training_data, int training_data_len, double learning_rate)
{
    Matrice **w;
    Matrice **b;
    Matrice **dnb;
    Matrice **dnw;
    try
        {
            w = new Matrice* [this->layers_num];
            b = new Matrice* [this->layers_num];
            dnb = new Matrice* [this->layers_num];
            dnw = new Matrice* [this->layers_num];
            for(int i = 0; i < this->layers_num; i++)
                {
                    int len = this->get_inputlen(i);
                    w[i] = new Matrice(this->layers[i], len);
                    dnw[i] = new Matrice(this->layers[i], len);
                    b[i] = new Matrice(this->layers[i], 1);
                    dnb[i] = new Matrice(this->layers[i], 1);
                    for(int j = 0; j < this->layers[i]; j++)
                        {
                            dnb[i]->data[j][0] = b[i]->data[j][0] = 0;
                            for(int k = 0; k < len; k++)
                                {
                                    dnw[i]->data[j][k] = w[i]->data[j][k] = 0;
                                }
                        }
                }
        }
    catch(bad_alloc& ba)
        {
            cerr<<"operator new failed"<<endl;
            return;
        }
    //print_mtx_list(dnw, 3);
    for(int i = 0; i < training_data_len; i++)
        {
            this->backpropagate(training_data[i], dnb, dnw);
            for(int j = 1; j < this->layers_num; j++)
                {
                    for(int k = 0; k < this->layers[j]; k++)
                        {
                            int len = this->get_inputlen(j);
                            b[j]->data[k][0] += dnb[j]->data[k][0];
                            for(int l = 0; l < len; l++)
                                {
                                    w[j]->data[k][l] += dnw[j]->data[k][l];
                                }
                        }
                }
        }
    //print_mtx_list(dnw, 3);
    double lr = learning_rate / training_data_len;
    for(int i = 1; i < this->layers_num; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    int len = this->get_inputlen(i);
                    this->biases[i]->data[j][0] -= lr * b[i]->data[j][0];
                    for(int k = 0; k < len; k++)
                        {
                            this->weights[i]->data[j][k] -= lr * w[i]->data[j][k];
                        }
                }
        }
    for(int i = 0; i < this->layers_num; i++)
        {
            delete w[i];
            delete dnw[i];
            delete b[i];
            delete dnb[i];
        }
    delete[] w;
    delete[] dnw;
    delete[] b;
    delete[] dnb;
}


void Network::test(MNIST_data **d)
{
    this->feedforward(d[8]->input);
    for(int i = 0; i < 10; i++)
        cout << this->outputs[this->layers_num - 1]->data[i][0] << "   ";
    cout << endl;
    for(int i = 0; i < 10; i++)
        {
            this->update_weights_and_biasses(&(d[i]), 100, 3);
            //cout << "set" << i << "done" << "\n";
        }
    this->feedforward(d[8]->input);
    for(int i = 0; i < 10; i++)
        cout << this->outputs[this->layers_num - 1]->data[i][0] << "   ";
    cout << '\n';
    for(int i = 0; i < 10; i++)
        cout << d[8]->required_output[i][0] << "   ";
    cout << '\n';
}

Matrice Network::get_output(double **input)
{
    this->feedforward(input);
    Matrice ret = this->outputs[this->layers_num - 1][0];
    return ret;
}
