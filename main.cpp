#include <iostream>
#include <fstream>
#include "MNIST_data.h"
#include "network.h"
#include "matrice.h"
#include "random.h"
#include <iosfwd>

using namespace std;


int main()
{
    ifstream input, required_output, validation_input, validation_output;
    input.open("/home/andrej/hdd/dokumentumok/neural_networks/data/training_data/input.dat", ios::in|ios::binary);
    required_output.open("/home/andrej/hdd/dokumentumok/neural_networks/data/training_data/required_output.dat", ios::in|ios::binary);
    validation_input.open("/home/andrej/hdd/dokumentumok/neural_networks/data/training_data/validation_input.dat", ios::in|ios::binary);
    validation_output.open("/home/andrej/hdd/dokumentumok/neural_networks/data/training_data/validation_output.dat", ios::in|ios::binary);
    MNIST_data *m[50000];
    MNIST_data *validation[10000];
    for(int i = 0; i < 50000; i++)
        {
            m[i] = new MNIST_data(784, 10);
            m[i]->load_data(input, required_output);
        }
    for(int i = 0; i < 10000; i++)
        {
            validation[i] = new MNIST_data(784, 1);
            validation[i]->load_data(validation_input, validation_output);
            //cout << validation[i]->required_output[0][0];
        }
    cout << "Hello world!" << endl;
    LayerDescriptor *layers[2];
    layers[0] = new LayerDescriptor(FULLY_CONNECTED, 30, SIGMOID);
    layers[1] = new LayerDescriptor(FULLY_CONNECTED, 10, SIGMOID);
    Network n(2, layers, 784);
    n.test(m, validation);
    input.close();
    required_output.close();
    /*input.open("/dev/urandom", ios::in);
    int i = random(0, 20, input);
    cout << i;*/
    return 0;
}
