#include "network.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "random.h"

using namespace std;

int getmax(double **d);

Network::Network(int layers_num, int *layers, int costfunction_type,  bool dropout, int neuron_type):
                biases(NULL), outputs(NULL), weights(NULL), neuron(neuron_type), dropout(dropout)
{
    try
    {
        this->costfunction_type = costfunction_type;
        this->total_layers_num = layers_num;
        this->neuron_type = neuron_type;
        this->layers_num = layers_num - 1;
        this->layers = new int [layers_num];
        for(int i = 0; i < layers_num; i++)
            this->layers[i] = layers[i];
        this->layers += 1;
        this->biases = new Matrice* [this->layers_num];
        this->outputs = new Matrice* [this->total_layers_num];
        this->weights = new Matrice* [this->layers_num];
        for(int i = 0; i < this->layers_num; i++)
            {
                this->biases[i] = new Matrice(this->layers[i], 1);
                this->outputs[i] = new Matrice(this->layers[i - 1], 1);
                this->weights[i] = new Matrice(this->layers[i], this->layers[i - 1]);
            }
        this->outputs[this->layers_num] = new Matrice(this->layers[this->layers_num - 1], 1);
        this->outputs += 1;
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
    this->outputs -= 1;
    for(int i = 0; i < layers_num; i++)
        {
            delete this->biases[i];
            delete this->outputs[i];
            delete this->weights[i];
        }
    delete this->biases;
    delete this->outputs;
    delete this->weights;
    this->layers -= 1;
    delete this->layers;
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
                    for(int k = 0; k < this->layers[i - 1]; k++)
                        {
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
    for(int i = 0; i < this->layers[layer]; i++)
        {
            this->outputs[layer]->data[i][0] = this->neuron.neuron(this->weights[layer]->data[i], input, this->biases[layer]->data[i][0], this->layers[layer - 1]);
        }
}

inline void Network::feedforward(double **input)
{
    for(int i = 0; i < this->layers[-1]; i++)
        {
            this->outputs[-1]->data[i][0] = input[i][0];
        }
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers_output(this->outputs[i - 1]->data, i);
        }
}

inline Matrice Network::get_delta(double **output, double **required_output)
{
    Matrice mtx(this->layers[this->layers_num - 1], 1);
    Matrice delta(this->layers[this->layers_num - 1], 1);
    Matrice output_derivate(this->layers[this->layers_num - 1], 1);
    switch(this->costfunction_type)
        {
        case QUADRATIC_CF:
            for(int i = 0; i < this->layers[this->layers_num - 1]; i++)
                {
                    mtx.data[i][0] = output[i][0] - required_output[i][0];
                }
            output_derivate = this->derivate_layers_output(this->layers_num - 1, this->outputs[this->layers_num - 2]->data);
            delta = hadamart_product(mtx, output_derivate);
            return delta;
        case CROSS_ENTROPY_CF:
            switch(this->neuron_type)
                {
                case SIGMOID:
                    for(int i = 0; i < this->layers[this->layers_num - 1]; i++)
                        {
                            mtx.data[i][0] = output[i][0] - required_output[i][0];
                        }
                    return mtx;
                default:
                    output_derivate = this->derivate_layers_output(this->layers_num - 1, this->outputs[this->layers_num - 2]->data);
                    for(int i = 0; i < this->layers[this->layers_num - 1]; i++)
                        {
                            delta.data[i][0] = (output_derivate.data[i][0] * (this->outputs[this->layers_num - 1]->data[i][0] - required_output[i][0])) /
                                                    (this->outputs[this->layers_num - 1]->data[i][0] * (1 - this->outputs[this->layers_num - 1]->data[i][0]));
                        }
                    return delta;
                }
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        }
}

inline void Network::backpropagate(MNIST_data *trainig_data, Matrice **nabla_b, Matrice **nabla_w)
{
    Matrice multiplied, output_derivate;
    this->feedforward(trainig_data->input);
    Matrice delta = this->get_delta(this->outputs[this->layers_num - 1]->data, trainig_data->required_output);
    *(nabla_b[this->layers_num - 1]) = delta;
    *(nabla_w[this->layers_num - 1]) = delta * this->outputs[this->layers_num - 2]->transpose();
    /*passing backwards the error*/
    for(int i = this->layers_num - 2; i >= 0; i--)
        {
            output_derivate = this->derivate_layers_output(i, this->outputs[i - 1]->data);
            multiplied = this->weights[i + 1]->transpose() * delta;
            delta = hadamart_product(multiplied, output_derivate);
            *(nabla_b[i]) = delta;
            *(nabla_w[i]) = delta * outputs[i -1]->transpose();
        }
}

inline Matrice Network::derivate_layers_output(int layer, double **input)
{
    Matrice mtx(this->layers[layer], 1);
    for(int i = 0; i < this->layers[layer]; i++)
        {
            mtx.data[i][0] = this->neuron.neuron_derivate(this->weights[layer]->data[i], input, this->biases[layer]->data[i][0], this->layers[layer - 1]);
        }
    return mtx;
}


void Network::update_weights_and_biasses(MNIST_data **training_data, int training_data_len, int total_trainingdata_len, double learning_rate, double regularization_rate)
{
    Matrice **w;
    Matrice **b;
    Matrice **dnb;
    Matrice **dnw;
    Matrice **b_bck, **w_bck;
    int *layer_bck, **ind;
    this->remove_some_neurons(&w_bck, &b_bck, &layer_bck, &ind);
    try
        {
            w = new Matrice* [this->layers_num];
            b = new Matrice* [this->layers_num];
            dnb = new Matrice* [this->layers_num];
            dnw = new Matrice* [this->layers_num];
            for(int i = 0; i < this->layers_num; i++)
                {
                    w[i] = new Matrice(this->layers[i], this->layers[i - 1]);
                    dnw[i] = new Matrice(this->layers[i], this->layers[i - 1]);
                    b[i] = new Matrice(this->layers[i], 1);
                    dnb[i] = new Matrice(this->layers[i], 1);
                    for(int j = 0; j < this->layers[i]; j++)
                        {
                            dnb[i]->data[j][0] = b[i]->data[j][0] = 0;
                            for(int k = 0; k < this->layers[i - 1]; k++)
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
    for(int i = 0; i < training_data_len; i++)
        {
            this->backpropagate(training_data[i], dnb, dnw);
            for(int j = 0; j < this->layers_num; j++)
                {
                    for(int k = 0; k < this->layers[j]; k++)
                        {
                            b[j]->data[k][0] += dnb[j]->data[k][0];
                            for(int l = 0; l < this->layers[j - 1]; l++)
                                {
                                    w[j]->data[k][l] += dnw[j]->data[k][l];
                                }
                        }
                }
        }
    double lr = learning_rate / training_data_len;
    double reg = (1 - learning_rate * (regularization_rate / total_trainingdata_len));
    for(int i = 0; i < this->layers_num; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    this->biases[i]->data[j][0] -= lr * b[i]->data[j][0];
                    for(int k = 0; k < this->layers[i - 1]; k++)
                        {
                            this->weights[i]->data[j][k] = reg * this->weights[i]->data[j][k] - lr * w[i]->data[j][k];
                        }
                }
        }
    this->add_back_removed_neurons(w_bck, b_bck, layer_bck, ind);
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

void Network::stochastic_gradient_descent(MNIST_data **training_data, int epochs, int epoch_len, double learning_rate,
                                            double regularization_rate, MNIST_data **test_data, int test_data_len, int trainingdata_len)
{
    MNIST_data **minibatch = new MNIST_data* [epoch_len];
    ifstream rand;
    rand.open("/dev/urandom", ios::in);
    int learning_accuracy;
    Matrice output;
    for(int i = 0; i < epochs; i++)
        {
            for(int j = 0; j < epoch_len; j++)
                {
                    minibatch[j] = training_data[random(0, trainingdata_len, rand)];
                }
            this->update_weights_and_biasses(minibatch, epoch_len, trainingdata_len, learning_rate, regularization_rate);
            if(test_data != NULL)
                {
                    learning_accuracy = 0;
                    for(int j = 0; j < 10000; j++)
                        {
                            output = this->get_output(test_data[j]->input);
                            if(getmax(output.data) == getmax(test_data[j]->required_output))
                                {
                                    learning_accuracy++;
                                }
                        }
                    cout << "set " << i << ": " << learning_accuracy << endl;
                }
        }
    rand.close();
}

int getmax(double **d)
{
    double Max = d[0][0];
    int index = 0;
    for(int i = 0; i < 10; i++)
        {
            if(Max < d[i][0])
                {
                    Max = d[i][0];
                    index = i;
                }
        }
    return index;
}
void Network::test(MNIST_data **d)
{
    this->stochastic_gradient_descent(d, 30, 100, 3, 5, d);
}

Matrice Network::get_output(double **input)
{
    this->feedforward(input);
    Matrice ret = this->outputs[this->layers_num - 1][0];
    return ret;
}
int partition( int *a, int l, int r) {
   int pivot, i, j, t;
   pivot = a[l];
   i = l; j = r+1;

   while( 1)
   {
   	do ++i; while( a[i] <= pivot && i <= r );
   	do --j; while( a[j] > pivot );
   	if( i >= j ) break;
   	t = a[i]; a[i] = a[j]; a[j] = t;
   }
   t = a[l]; a[l] = a[j]; a[j] = t;
   return j;
}

void quickSort( int *a, int l, int r)
{
   int j;

   if( l < r )
   {
   	// divide and conquer
        j = partition( a, l, r);
       quickSort( a, l, j-1);
       quickSort( a, j+1, r);
   }

}

inline void Network::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ///TAKE CARE!!! if this->dropout == false the function must return immediatelly!!!
    if((this->total_layers_num <= 2) || (this->dropout == false))
        return;
    ifstream rand;
    rand.open("/dev/urandom", ios::in);
    layers_bckup[0] = new int [this->total_layers_num];
    this->layers -= 1;
    for(int i = 0; i < this->total_layers_num; i++)
        layers_bckup[0][i] = this->layers[i];
    for(int i = 1; i < this->layers_num; i++)
        this->layers[i] >>= 1;
    this->layers += 1;
    layers_bckup[0] += 1;
    w_bckup[0] = new Matrice* [this->total_layers_num - 1];
    b_bckup[0] = new Matrice* [this->total_layers_num - 1];
    for(int i = 0; i < this->layers_num; i++)
        {
            w_bckup[0][i] = this->weights[i];
            b_bckup[0][i] = this->biases[i];
            this->biases[i] = new Matrice(this->layers[i], 1);
            this->weights[i] = new Matrice(this->layers[i], this->layers[i - 1]);
        }
    indexes[0] = new int* [this->total_layers_num - 2];
    int *tmp;
    for(int i = 0; i < this->total_layers_num - 2; i++)
        {
            indexes[0][i] = new int [this->layers[i]];
            tmp = new int[layers_bckup[0][i]];
            for(int j = 0; j < layers_bckup[0][i]; j++)
                tmp[j] = j;
            shuffle(tmp, layers_bckup[0][i], rand);
            for(int j = 0; j < this->layers[i]; j++)
                {
                    indexes[0][i][j] = tmp[j];
                }
            quickSort(indexes[0][i], 0, this->layers[i] - 1);
            delete[] tmp;
        }
    for(int j = 0; j < this->layers[0]; j++)
        {
            this->biases[0]->data[j][0] = b_bckup[0][0]->data[indexes[0][0][j]][0];
            for(int k = 0; k < this->layers[-1]; k++)
                {
                    this->weights[0]->data[j][k] = w_bckup[0][0]->data[indexes[0][0][j]][k];
                }
        }
    for(int i = 1; i < this->layers_num - 1; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    this->biases[i]->data[j][0] = b_bckup[0][i]->data[indexes[0][i][j]][0];
                    for(int k = 0; k < this->layers[i - 1]; k++)
                        {
                            this->weights[i]->data[j][k] = w_bckup[0][i]->data[indexes[0][i][j]][indexes[0][i - 1][k]];
                        }
                }
        }
    for(int j = 0; j < this->layers[this->layers_num - 1]; j++)
        {
            this->biases[this->layers_num - 1]->data[j][0] = b_bckup[0][this->layers_num - 1]->data[j][0];
            for(int k = 0; k < this->layers[this->layers_num - 2]; k++)
                {
                    this->weights[this->layers_num - 1]->data[j][k] = w_bckup[0][this->layers_num - 1]->data[j][indexes[0][this->layers_num - 2][k]];
                }
        }
    rand.close();
}

inline void Network::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ///TAKE CARE!!! if this->dropout == false the function must return immediatelly!!!
    if((this->total_layers_num <= 2) || (this->dropout == false))
        return;
    for(int j = 0; j < this->layers[0]; j++)
        {
            b_bckup[0]->data[indexes[0][j]][0] = this->biases[0]->data[j][0];
            for(int k = 0; k < this->layers[-1]; k++)
                {
                    w_bckup[0]->data[indexes[0][j]][k] = this->weights[0]->data[j][k];
                }
        }
    for(int i = 1; i < this->layers_num - 1; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    b_bckup[i]->data[indexes[i][j]][0] = this->biases[i]->data[j][0];
                    for(int k = 0; k < this->layers[i - 1]; k++)
                        {
                            w_bckup[i]->data[indexes[i][j]][indexes[i][k]] = this->weights[i]->data[j][k];
                        }
                }
        }
    for(int j = 0; j < this->layers[this->layers_num - 1]; j++)
        {
            b_bckup[this->layers_num - 1]->data[j][0] = this->biases[this->layers_num - 1]->data[j][0];
            for(int k = 0; k < this->layers[this->layers_num - 2]; k++)
                {
                    w_bckup[this->layers_num - 1]->data[j][indexes[this->layers_num - 2][k]] = this->weights[this->layers_num - 1]->data[j][k];
                }
        }
    for(int i = 0; i < this->layers_num; i++)
        {
            delete this->biases[i];
            this->biases[i] = b_bckup[i];
            delete this->weights[i];
            this->weights[i] = w_bckup[i];
            if(i < (this->layers_num - 1))
                delete[] indexes[i];
        }
    delete[] indexes;
    this->layers -= 1;
    layers_bckup -= 1;
    for(int i = 0; i < this->total_layers_num; i++)
        this->layers[i] = layers_bckup[i];
    this->layers += 1;
    delete[] layers_bckup;

}
