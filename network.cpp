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
                dropout(dropout), input_row(input_row), input_col(input_col), input_channel_count(input_channel_count), layers_num(layers_num),
                costfunction_type(costfunction_type), openclenv(), nabla(NULL), deltanabla(NULL)
{
    try
    {
        this->total_layers_num = layers_num + 1;
        this->construct_layers(layerdesc);
    }
    catch(bad_alloc &ba)
        {
            cerr << "bad alloc in Network::constructor" << endl;
        }
}

Network::Network(char *data): openclenv(), nabla(NULL), deltanabla(NULL)
{
    ///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout
    ///int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1
    ifstream file (data, ios::in|ios::binary);
    if(file.is_open())
        {
            int f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_stride;
            LayerDescriptor *dsc[this->layers_num];
            file.read(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_col)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->costfunction_type)), sizeof(int));
            file.read(reinterpret_cast<char *>(&(this->dropout)), sizeof(bool));
            this->total_layers_num = this->layers_num + 1;
            for(int i = 0; i < this->layers_num; i++)
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
    if(layerdesc[0][0].layer_type == FULLY_CONNECTED)
        this->layers[0] = new InputLayer(input_row, 1, 1, SIGMOID, p, FULLY_CONNECTED, &(this->openclenv));
    else
        this->layers[0] = new InputLayer(input_row, input_col, input_channel_count, SIGMOID, p, CONVOLUTIONAL, &(this->openclenv));
    this->layers += 1;
    for(int i = 0; i < layers_num; i++)
    {
        this->layerdsc[i] = new LayerDescriptor(layerdesc[i][0].layer_type, layerdesc[i][0].neuron_type, layerdesc[i][0].neuron_count,
                                                layerdesc[i][0].col, layerdesc[i][0].mapcount, layerdesc[i][0].stride);
        switch(layerdesc[i][0].layer_type)
        {
            case FULLY_CONNECTED:
                this->layers[i] = new FullyConnected(layerdesc[i][0].neuron_count, this->layers[i - 1][0].get_output_len(),
                                                     layerdesc[i][0].neuron_type, &(this->openclenv));
                break;
            /*case SOFTMAX:
                this->layers[i] = new Softmax(layerdesc[i][0].neuron_count, this->layers[i - 1][0].get_output_len());
                break;
            case CONVOLUTIONAL:
                ///Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int stride=1)
                this->layers[i] = new Convolutional(this->layers[i - 1][0].get_output_row(), this->layers[i - 1][0].get_output_col(),
                                                    this->layers[i - 1][0].get_mapcount(), layerdesc[i][0].row, layerdesc[i][0].col,
                                                    layerdesc[i][0].mapcount, SIGMOID, layerdesc[i + 1][0].layer_type, p, 1);
                break;*/
            default:
                cerr << "Unknown layer type: " << layerdesc[i][0].layer_type << "\n";
                throw std::exception();
        }
    }
}

inline void Network::feedforward(MatrixData **input)
{
    this->layers[-1][0].set_input(input);
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers[i][0].layers_output(this->layers[i - 1][0].get_output());
        }
}

double Network::cost(MatrixData &required_output, int req_outp_indx)
{
    double helper = 0, result = 0;
    this->layers[this->layers_num - 1][0].sync_memory();
    switch(this->costfunction_type)
        {
        case QUADRATIC_CF:
            /// 1/2 * ||y(x) - a||^2
            for(int i = 0; i < this->layers[this->layers_num - 1][0].get_output_len(); i++)
                {
                    helper = required_output[i][0] - (this->layers[this->layers_num - 1][0].get_output()[0][0])[i][0];
                    result += helper * helper;
                }
            return 0.5 * result;
        case CROSS_ENTROPY_CF:
            ///y(x)ln a + (1 - y(x))ln(1 - a)
            for(int i = 0; i < this->layers[this->layers_num - 1][0].get_output_len(); i++)
                {
                    helper += required_output[i][0] * log((this->layers[this->layers_num - 1][0].get_output()[0][0])[i][0]) + (1 - required_output[i][0]) *
                                    log(1 - (this->layers[this->layers_num - 1][0].get_output()[0][0])[i][0]);
                }
            return helper;
        case LOG_LIKELIHOOD_CF:
            result = -1 * log((this->layers[this->layers_num - 1][0].get_output()[0][0])[req_outp_indx][0]);
            return result;
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        }
}


inline void Network::backpropagate(MNIST_data *trainig_data, Layers_features **nabla)
{
    ///currently the final layer has to be a clasification layer
    this->feedforward(trainig_data[0].input);
    MatrixData **delta;
    cl_event event[2];
    delta = this->layers[layers_num - 1][0].get_output_error(this->layers[layers_num - 2][0].get_output(),
                                                    trainig_data[0].required_output, this->costfunction_type);
    ///nabla[this->layers_num - 1][0].fmap[0][0].biases[0][0] = delta[0][0];
    clEnqueueCopyBuffer(nabla[this->layers_num - 1][0].fmap[0][0].mtxop[0].command_queue, delta[0][0].cl_mem_obj, nabla[this->layers_num - 1][0].fmap[0][0].biases[0][0].cl_mem_obj,
                        0, 0, nabla[this->layers_num - 1][0].fmap[0][0].biases[0][0].get_row(), 0, NULL, &event[0]);
    ///nabla[this->layers_num - 1][0].fmap[0][0].weights[0][0] = delta[0][0] * this->layers[this->layers_num - 2][0].get_output()[0][0].transpose();
    nabla[this->layers_num - 1][0].fmap[0][0].mtxop[0].multiply_with_transpose(delta[0][0], this->layers[this->layers_num - 2][0].get_output()[0][0],
                                                                               nabla[this->layers_num - 1][0].fmap[0][0].weights[0][0], 0, NULL, &event[1]);
    clWaitForEvents(2, event);
    //print_mtx(delta[0][0], &(nabla[this->layers_num - 1][0].fmap[0][0].mtxop[0].command_queue));
    ///*passing backwards the error*/
    for(int i = this->layers_num - 2; i >= 0; i--)
        {
            delta = this->layers[i][0].backpropagate(this->layers[i - 1][0].get_output(),
                                           this->layers[i + 1][0].get_feature_maps(), nabla[i][0].fmap, delta,
                                           nabla[i+1][0].get_fmap_count());
            //print_mtx(delta[0][0], &(nabla[this->layers_num - 1][0].fmap[0][0].mtxop[0].command_queue));
        }
    //throw exception();
    /*if(this->layers[0][0].get_mapcount() > 1)
        {
            for(int i = 0; i < this->layers[0][0].get_mapcount(); i++)
                delete delta[i];
            delete[] delta;
        }
    else
        {
            delete delta[0];
            delete[] delta;
        }*/
}

void Network::update_weights_and_biasses(MNIST_data **training_data, int training_data_len, int total_trainingdata_len, double learning_rate, double regularization_rate)
{
    int *layer_bck, **ind;
    //this->remove_some_neurons(&w_bck, &b_bck, &layer_bck, &ind);
    if(this->nabla == NULL)
    {
        try
        {
            this->nabla = new Layers_features* [this->layers_num];
            this->deltanabla = new Layers_features* [this->layers_num];
            for(int i = 0; i < this->layers_num; i++)
            {
                ///Layers_features(int mapcount, int row, int col, int depth, int biascnt);
                int biascnt;
                if((this->layers[i][0].get_layer_type() == FULLY_CONNECTED) or (this->layers[i][0].get_layer_type() == SOFTMAX))
                    biascnt = this->layers[i][0].get_weights_row();
                else
                    biascnt = 1;
                this->nabla[i] = new Layers_features(this->layers[i][0].get_mapcount(),
                                               this->layers[i][0].get_weights_row(),
                                               this->layers[i][0].get_weights_col(),
                                               this->layers[i][0].get_mapdepth(),
                                               biascnt,
                                               &(this->openclenv));
                this->deltanabla[i] = new Layers_features(this->layers[i][0].get_mapcount(),
                                                    this->layers[i][0].get_weights_row(),
                                                    this->layers[i][0].get_weights_col(),
                                                    this->layers[i][0].get_mapdepth(),
                                                    biascnt,
                                                    &(this->openclenv));
            }
        }
        catch(bad_alloc& ba)
        {
            cerr<<"operator new failed in the function: Network::update_weights_and_biasses"<<endl;
            return;
        }
    }
    else
    {
        cl_event events[4];
        for(int i=0; i<this->layers_num; i++)
        {
            for(int j=0; j<this->nabla[i][0].get_fmap_count();j++)
            {
                for(int k = 0; k < this->nabla[i][0].fmap[j][0].get_mapdepth(); k++)
                {
                    this->nabla[i][0].fmap[0][0].mtxop[0].zero(this->nabla[i][0].fmap[j][0].biases[k][0], 0, NULL, &events[0]);
                    this->nabla[i][0].fmap[0][0].mtxop[0].zero(this->nabla[i][0].fmap[j][0].weights[k][0], 0, NULL, &events[1]);
                    this->deltanabla[i][0].fmap[0][0].mtxop[0].zero(this->nabla[i][0].fmap[j][0].biases[k][0], 0, NULL, &events[2]);
                    this->deltanabla[i][0].fmap[0][0].mtxop[0].zero(this->nabla[i][0].fmap[j][0].weights[k][0], 0, NULL, &events[3]);
                    clWaitForEvents(4, events);
                }
            }
        }
    }
    for(int i = 0; i < training_data_len; i++)
        {
            this->backpropagate(training_data[i], deltanabla);
            for(int j = 0; j < this->layers_num; j++)
                {
                    nabla[j][0] += deltanabla[j][0];
                }
        }
    double lr = learning_rate / training_data_len;
    double reg = (1 - learning_rate * (regularization_rate / total_trainingdata_len));
    for(int i = 0; i < this->layers_num; i++)
        {
            this->layers[i][0].update_weights_and_biasses(lr, reg, nabla[i]);
        }
}

MatrixData Network::get_output(MatrixData **input)
{
    ///TODO modify this function to work with multiple input features...
    this->feedforward(input);
    this->layers[this->layers_num - 1][0].sync_memory();
    MatrixData ret = this->layers[this->layers_num - 1][0].get_output()[0][0];
    return ret;
}

inline void Network::remove_some_neurons(MatrixData ***w_bckup, MatrixData ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void Network::add_back_removed_neurons(MatrixData **w_bckup, MatrixData **b_bckup, int *layers_bckup, int **indexes)
{
    ;
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
            network_params.write(reinterpret_cast<char *>(&(this->dropout)), sizeof(bool));
            for(int i=0; i<this->layers_num; i++)
                {
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i][0].layer_type)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i][0].neuron_type)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i][0].neuron_count)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i][0].col)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i][0].mapcount)), sizeof(int));
                    network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i][0].stride)), sizeof(int));
                }
            for(int i = -1; i < this->layers_num; i++)
                {
                    this->layers[i][0].store(network_params);
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
    MatrixData helper(this->layers[this->layers_num - 1][0].get_output_row(), 1);
    for(int i = 0; i < this->layers[this->layers_num - 1][0].get_output_row(); i++)
        {
            helper[i][0] == 0;
        }
    MatrixData output;
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
                            output = this->get_output(test_data[j][0].input);
                            if(getmax(output.data) == (test_data[j][0].required_output)[0][0])
                                {
                                    learning_accuracy++;
                                }
                            if(monitor_learning_cost)
                                {
                                    //if(j > 0)
                                    //    helper[(int)test_data[j - 1]->required_output[0][0]][0] = 0;
                                    helper[(int)test_data[j][0].required_output[0][0]][0] = 1;
                                    learning_cost += this->cost(helper, test_data[j][0].required_output[0][0]);
                                    helper[(int)test_data[j][0].required_output[0][0]][0] = 0;
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
    MatrixData helper(this->layers[this->layers_num - 1][0].get_output_row(), 1);
    MatrixData output;
    int test_data_len = 10000;
    bool monitor_learning_cost = true;
    for(int i = 0; i < this->layers[this->layers_num - 1][0].get_output_row(); i++)
        {
            helper[i][0] == 0;
        }
    learning_accuracy = learning_cost = 0;
    for(int j = 0; j < test_data_len; j++)
        {
            ///TODO this is an errorprone as well
            output = this->get_output(test_data[j][0].input);
            if(getmax(output.data) == test_data[j][0].required_output[0][0])
                {
                    learning_accuracy++;
                }
            if(monitor_learning_cost)
                {
                    //if(j > 0)
                    //    helper[(int)test_data[j - 1]->required_output[0][0]][0] = 0;
                    helper[(int)test_data[j][0].required_output[0][0]][0] = 1;
                    learning_cost += this->cost(helper, test_data[j][0].required_output[0][0]);
                    helper[(int)test_data[j][0].required_output[0][0]][0] = 0;
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
    ///(training_data, epochs, minibatch_len, learning_rate, monitor_learning_cost, regularization_rate, test_data, minibatch_count, test_data_len, trainingdata_len)
    this->stochastic_gradient_descent(d, 3, 10, 0.03, true, 1, v, -1);
    /*for(int i=0; i<30;i++)
    {
        this->check_accuracy(v);
        cout << i << endl;
    }*/
    //this->store("/home/andrej/myfiles/Asztal/net.bin");
    //MatrixData o = this->get_output(v[0]->input);
    //print_mtx(o);
}
