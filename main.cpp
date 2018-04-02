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
    input.open("/home/andrej/myfiles/dokumentumok/neural_networks/data/training_data/input.dat", ios::in|ios::binary);
    required_output.open("/home/andrej/myfiles/dokumentumok/neural_networks/data/training_data/required_output.dat", ios::in|ios::binary);
    validation_input.open("/home/andrej/myfiles/dokumentumok/neural_networks/data/training_data/validation_input.dat", ios::in|ios::binary);
    validation_output.open("/home/andrej/myfiles/dokumentumok/neural_networks/data/training_data/validation_output.dat", ios::in|ios::binary);
    cout << "the required files are opened\n";
    MNIST_data *m[50000];
    MNIST_data *validation[10000];
    for(int i = 0; i < 50000; i++)
        {
            m[i] = new MNIST_data(784, 1, 10, 1);
            m[i]->load_data(input, required_output);
        }
    for(int i = 0; i < 10000; i++)
        {
            validation[i] = new MNIST_data(784, 1, 1, 1);
            validation[i]->load_data(validation_input, validation_output);
            //cout << validation[i]->required_output[0][0];
        }
    cout << "the training data and the validation data is loaded\n";
    LayerDescriptor *layers[4];
    layers[0] = new LayerDescriptor(CONVOLUTIONAL, SIGMOID, 4, 6, 5);
    layers[1] = new LayerDescriptor(CONVOLUTIONAL, SIGMOID, 7, 5, 5);
    layers[2] = new LayerDescriptor(FULLY_CONNECTED, SIGMOID, 30);
    layers[3] = new LayerDescriptor(SOFTMAX, SIGMOID, 10);
    //Network n(2, layers, 784, 1, 1, LOG_LIKELIHOOD_CF);
    Network n(4, layers, 28, 28, 1, LOG_LIKELIHOOD_CF);
    n.test(m, validation);
    input.close();
    required_output.close();
    /*double inp[5][5] = {{2,7,5,3,6},{4,9,8,6,3},{7,5,4,6,8},{1,0,7,4,2},{9,4,5,6,1}};
    Matrice input(5,5);
    for(int i = 0; i<5; i++)
        for(int j = 0; j<5; j++)
            input.data[i][j] = inp[i][j];
    double kern[3][3] = {{4,7,2},{1,5,9},{8,5,2}};
    Matrice kernel(3,3);
    for(int i = 0; i<3; i++)
        for(int j = 0; j<3; j++)
            kernel.data[i][j] = kern[i][j];
    Matrice output1(3,3);
    Matrice output2(3,3);
    convolution(input,kernel,output1);
    print_mtx(output1);
    conv(input,kernel,output2);
    print_mtx(output2);*/
    return 0;
}
