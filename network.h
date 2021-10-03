#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "neurons/neuron.h"
#include "data_loader/MNIST_data.h"
#include "matrix/matrix.h"
#include "layers/layers.h"
//#include "SGD.h"


class Network{
    int total_layers_num, layers_num, input_row, input_col, input_channel_count, threadcount;
    Layer **layers;
    LayerDescriptor **layerdsc;
    void construct_layers(LayerDescriptor **desc);
    void feedforward(Matrix **input, int threadindex=0);
    public:
    //double dropout_probability;
    void backpropagate(MNIST_data *training_data, Layers_features **nabla, int costfunction_type, int threadindex=0);
    void store(char *filename);
    Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int input_channel_count = 1);//,
            //int costfunction_type = CROSS_ENTROPY_CF, double dropout_probability = 0);
    Network(char *data);
    ~Network();
    Matrix get_output(Matrix **input, int threadindex=0);
    friend class StochasticGradientDescent;
    friend class Job;
    void set_threadcount(int threadcnt);
    int get_threadcount();
};

#endif // NETWORK_H_INCLUDED
