#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED
#include "neuron.h"
#include "MNIST_data.h"
#include "matrice.h"
#include "layers/layers.h"

class Network{
    int total_layers_num, layers_num, costfunction_type, input_row, input_col, input_channel_count;
    Layer **layers;
    bool dropout;
    inline void backpropagate(MNIST_data *training_data, Layers_features **nabla);
    void update_weights_and_biasses(MNIST_data **training_data, int training_data_len, int total_trainingdata_len, double learning_rate, double regularization_rate);
    inline void remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes);
    inline void add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes);
    inline void feedforward(Matrice **input);
    double cost(Matrice &required_output);
    public:
    void load(char *filename);
    void store(char *filename);
    void stochastic_gradient_descent(MNIST_data **training_data, int epochs, int epoch_len, double learning_rate, bool monitor_learning_cost = false,
                                    double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);
    Network(int layers_num, LayerDescriptor **layerdesc, int input_row, int input_col = 1, int input_channel_count = 1,
            int costfunction_type = CROSS_ENTROPY_CF, bool dropout = false);
    ~Network();
    void test(MNIST_data **d, MNIST_data **v);
    Matrice get_output(Matrice **input);
};

#endif // NETWORK_H_INCLUDED
