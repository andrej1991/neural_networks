#include <iostream>
#include <fstream>
#include "MNIST_data.h"
#include "network.h"
#include "matrix/matrix.h"
#include "random.h"
#include <iosfwd>

using namespace std;

#define InputRow 784
#define InputCol 1

#include "opencl_setup.h"
#include "neuron/neuron.h"



int main()
{
    ifstream input, required_output, validation_input, validation_output;
    input.open("/home/andrej/hdd/Dokumentumok/neural_networks/data/training_data/input.dat", ios::in|ios::binary);
    required_output.open("/home/andrej/hdd/Dokumentumok/neural_networks/data/training_data/required_output.dat", ios::in|ios::binary);
    validation_input.open("/home/andrej/hdd/Dokumentumok/neural_networks/data/training_data/validation_input.dat", ios::in|ios::binary);
    validation_output.open("/home/andrej/hdd/Dokumentumok/neural_networks/data/training_data/validation_output.dat", ios::in|ios::binary);
    cout << "the required files are opened\n";
    MNIST_data *m[50000];
    MNIST_data *validation[10000];
    for(int i = 0; i < 50000; i++)
        {
            m[i] = new MNIST_data(InputRow, InputCol, 10, 1);
            m[i]->load_data(input, required_output);
        }
    for(int i = 0; i < 10000; i++)
        {
            validation[i] = new MNIST_data(InputRow, InputCol, 1, 1);
            validation[i]->load_data(validation_input, validation_output);
            //cout << validation[i]->required_output[0][0];
        }
    cout << "the training data and the validation data is loaded\n";
    //OpenclSetup env;
    //Neuron n(&(env), SIGMOID);
    //n.neuron(m[0]->required_output);
    LayerDescriptor *layers[2];
    layers[0] = new LayerDescriptor(FULLY_CONNECTED, SIGMOID, 30);
    layers[1] = new LayerDescriptor(FULLY_CONNECTED, SIGMOID, 10);
    Network n(2, layers, InputRow, InputCol, 1, CROSS_ENTROPY_CF);
    //Network n("/home/andrej/myfiles/Asztal/net.bin");
    n.test(m, validation);
    input.close();
    required_output.close();
    cout << "Hello world\n";
    return 0;
}
