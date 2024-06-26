#include "network.h"
#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <math.h>
#include "layers/layers.h"
#include "layers/Convolutional.h"
#include "layers/Pooling.h"

using namespace std;
///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int costfunction_type = CROSS_ENTROPY_CF, bool dropout = false
Network::Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count):
                input_row(input_row), input_col(input_col), input_channel_count(input_channel_count), layers_num(layers_num){
    this->total_layers_num = layers_num + 1;
    this->threadcount = 1;
    this->deltas = new Matrix** [this->layers_num];
    try
    {
        this->construct_layers(layerdesc);
    }
    catch(bad_alloc &ba){
            cerr << "bad alloc in Network::constructor" << endl;
        }
}

Network::Network(char *data){
    ///int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col, int input_channel_count, int costfunction_type,  bool dropout
    ///int layer_type, int neuron_type, int neuron_count, int col = 1, int mapcount = 1, int stride = 1
    ifstream file (data, ios::in|ios::binary);
    if(file.is_open()){
        this->threadcount = 1;
        int _size, vec_size;
        char *name, *conn;
        vector<string> output_connections;
        int f_layer_type, f_neuron_type, f_neuron_count, f_col, f_mapcount, f_vertical_stride, f_horizontal_stride;
        file.read(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
        file.read(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
        file.read(reinterpret_cast<char *>(&(this->input_col )), sizeof(int));
        file.read(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
        this->total_layers_num = this->layers_num + 1;
        this->deltas = new Matrix** [this->layers_num];
        LayerDescriptor *dsc[this->layers_num];
        for(int i = 0; i < this->layers_num; i++){
            file.read(reinterpret_cast<char *>(&f_layer_type), sizeof(int));
            file.read(reinterpret_cast<char *>(&f_neuron_type), sizeof(int));
            file.read(reinterpret_cast<char *>(&f_neuron_count), sizeof(int));
            file.read(reinterpret_cast<char *>(&f_col), sizeof(int));
            file.read(reinterpret_cast<char *>(&f_mapcount), sizeof(int));
            file.read(reinterpret_cast<char *>(&f_vertical_stride), sizeof(int));
            file.read(reinterpret_cast<char *>(&f_horizontal_stride), sizeof(int));
            file.read(reinterpret_cast<char *>(&_size), sizeof(int));
            name = new char [_size];
            file.read(name, _size);
            file.read(reinterpret_cast<char *>(&vec_size), sizeof(int));
            for(int j = 0; j < vec_size; j++){
                file.read(reinterpret_cast<char *>(&_size), sizeof(int));
                conn = new char [_size];
                file.read(conn, _size);
                output_connections.push_back(string(conn));
                delete conn;
            }
            dsc[i] = new LayerDescriptor(f_layer_type, f_neuron_type, f_neuron_count, output_connections, string(name), f_col, f_mapcount, f_vertical_stride, f_horizontal_stride);
            delete name;
            output_connections.clear();
        }
        this->construct_layers(dsc);
        for(int i = 0; i < this->layers_num; i++){
            this->layers[i]->load(file);
        }
        file.close();
        for(int i = 0; i < this->layers_num; i++){
            delete dsc[i];
        }
    }
    else
    {
        cerr << "Unable to open the file:" << '"' << data << '"' << endl;
        throw exception();
    }
}

Network::~Network(){
    this->layers -= 1;
    for(int i = 0; i < this->layers_num; i++){
        delete this->layers[i];
        delete this->layerdsc[i];
    }
    delete this->layers[this->layers_num];
    delete[] this->layers;
    delete[] this->layerdsc;
}

vector<int> get_inputs(LayerDescriptor *layerdesc, map<string,int> &m){
    vector<int> inputs;
    for(string connection : layerdesc->get_input_connections()){
        inputs.push_back(m[connection]);
    }
    return inputs;
}

vector<int> get_output_connections(LayerDescriptor **layerdesc, map<string,int> &m, int layersnum, int indx){
    vector<int> outputs;
    for(int i = 0; i < layersnum; i++){
        for(string connection : layerdesc[i]->get_input_connections()){
            if(m[connection] == indx){
                outputs.push_back(i);
            }
        }
    }
    return outputs;
}

void Network::construct_layers(LayerDescriptor **layerdesc){
    this->layers = new Layer* [this->total_layers_num];
    this->layerdsc = new LayerDescriptor* [this->layers_num];
    Padding p;
    map<std::string, int> layer_name_to_index;
    vector<int> prev_outputlens;
    vector<int> inputs;
    vector<int> outputs;
    layer_name_to_index[string("InputLayer")] = -1;
    for(int i = 0; i < layers_num; i++){
        layer_name_to_index[layerdesc[i]->get_name()] = i;
    }
    if(layerdesc[0]->layer_type == FULLY_CONNECTED)
        this->layers[0] = new InputLayer(input_row, 1, 1, SIGMOID, p, FULLY_CONNECTED);
    else
        this->layers[0] = new InputLayer(input_row, input_col, input_channel_count, SIGMOID, p, CONVOLUTIONAL);
    this->layers += 1;
    for(int i = 0; i < layers_num; i++){
        this->layerdsc[i] = new LayerDescriptor(layerdesc[i]->layer_type, layerdesc[i]->neuron_type, layerdesc[i]->neuron_count, layerdesc[i]->input_connections, layerdesc[i]->name,
                                                layerdesc[i]->col, layerdesc[i]->mapcount, layerdesc[i]->vertical_stride, layerdesc[i]->horizontal_stride);
        inputs.clear();
        outputs.clear();
        inputs = get_inputs(layerdesc[i], layer_name_to_index);
        outputs = get_output_connections(layerdesc, layer_name_to_index, this->layers_num, i);
        switch(layerdesc[i]->layer_type){
            case FULLY_CONNECTED:
                this->layers[i] = new FullyConnected(layerdesc[i]->neuron_count, this->layers, inputs,
                                                     layerdesc[i]->neuron_type, i);
                break;
            case SOFTMAX:
                this->layers[i] = new Softmax(layerdesc[i]->neuron_count, this->layers, inputs, i);
                break;
            case FLATTEN:
                this->layers[i] = new Flatten(this->layers, i, inputs, layerdesc[outputs[0]]->row);
                break;
            case CONVOLUTIONAL:
                this->layers[i] = new Convolutional(this->layers, inputs, layerdesc[i]->row, layerdesc[i]->col,
                                                    layerdesc[i]->mapcount, layerdesc[i]->neuron_type, i, p, layerdesc[i]->vertical_stride, layerdesc[i]->horizontal_stride);
                break;
            case MAX_POOLING:
                this->layers[i] = new Pooling(this->layers, inputs, layerdesc[i]->row, layerdesc[i]->col, MAX_POOLING, i);
                break;
            default:
                cerr << "Unknown layer type: " << layerdesc[i]->layer_type << "\n";
                throw std::exception();
        }
        this->layers[i]->create_connections(inputs, outputs);
    }

    this->deltas[this->layers_num - 1] = new Matrix*;
}

void Network::store(char *filename){
    int _size, vec_size;
    const char *str;
    ofstream network_params (filename, ios::out | ios::binary);
    if(network_params.is_open()){
        network_params.write(reinterpret_cast<char *>(&(this->layers_num)), sizeof(int));
        network_params.write(reinterpret_cast<char *>(&(this->input_row)), sizeof(int));
        network_params.write(reinterpret_cast<char *>(&(this->input_col )), sizeof(int));
        network_params.write(reinterpret_cast<char *>(&(this->input_channel_count)), sizeof(int));
        for(int i=0; i<this->layers_num; i++){
            _size = this->layerdsc[i]->name.size();
            str = this->layerdsc[i]->name.c_str();
            vec_size = this->layerdsc[i]->input_connections.size();
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->layer_type)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->neuron_type)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->neuron_count)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->col)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->mapcount)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->vertical_stride)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(this->layerdsc[i]->horizontal_stride)), sizeof(int));
            network_params.write(reinterpret_cast<char *>(&(_size)), sizeof(int));
            network_params.write(str, this->layerdsc[i]->name.size());
            network_params.write(reinterpret_cast<char *>(&(vec_size)), sizeof(int));
            for(int j = 0; j < this->layerdsc[i]->input_connections.size(); j++){
                _size = this->layerdsc[i]->input_connections[j].size();
                str = this->layerdsc[i]->input_connections[j].c_str();
                network_params.write(reinterpret_cast<char *>(&(_size)), sizeof(int));
                network_params.write(str, this->layerdsc[i]->input_connections[j].size());
            }
        }
        for(int i = -1; i < this->layers_num; i++){
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

vector<Matrix***> Network::collect_inputs(int current_index, vector<int> inputs){
    vector<Matrix***> ret;
    Matrix ***retmatrix;
    for(int i : inputs){
        retmatrix = new Matrix** [this->threadcount];
        for(int j = 0; j < threadcount; j++){
            retmatrix[j] = this->layers[i]->get_output(j);
        }
        ret.push_back(retmatrix);
    }
    return ret;
}


void Network::feedforward(Matrix **input, int threadindex){
    this->layers[-1]->set_input(input, threadindex);
    for(int i = 0; i < this->layers_num; i++){
        this->layers[i]->layers_output(NULL, threadindex);
    }
}

Matrix Network::get_output(Matrix **input, int threadindex){
    ///TODO modify this function to work with multiple input features...
    this->feedforward(input, threadindex);
    Matrix ret = *(this->layers[this->layers_num - 1]->get_output(threadindex)[0]);
    return ret;
}

void Network::backpropagate(Data_Loader *trainig_data, Layers_features **nabla, int costfunction_type, int threadindex){
    ///currently the final layer has to be a clasification layer
    this->feedforward(trainig_data->input, threadindex);
    this->deltas[layers_num - 1][0] = this->layers[layers_num - 1]->get_output_error(this->layers[layers_num - 2]->get_output(threadindex),
                                                    trainig_data->required_output, costfunction_type, threadindex);
    nabla[this->layers_num - 1]->fmap[0]->biases[0][0] = this->deltas[layers_num - 1][0][0];
    nabla[this->layers_num - 1]->fmap[0]->weights[0][0] = this->deltas[layers_num - 1][0][0] * this->layers[this->layers_num - 2]->get_output(threadindex)[0]->transpose();
    for(int i = this->layers_num - 2; i >= 0; i--){
        this->layers[i]->backpropagate(this->layers[i - 1]->get_output(threadindex), this->layers[i + 1], nabla[i][0].fmap, this->deltas, threadindex);
    }
}

int Network::get_threadcount(){
    return this->threadcount;
}

void Network::set_threadcount(int threadcnt){
    if(this->threadcount != threadcnt){
        this->layers[-1]->set_threadcount(threadcnt);
        for(int i = 0; i < this->layers_num; i++){
            this->layers[i]->set_threadcount(threadcnt);
        }
        this->threadcount = threadcnt;
    }
    cout << "threadcounts set\n";
}

