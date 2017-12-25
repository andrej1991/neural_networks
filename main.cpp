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
    LayerDescriptor *layers[2];
    layers[0] = new LayerDescriptor(CONVOLUTIONAL, SIGMOID, 5, 5, 4);
    //layers[0] = new LayerDescriptor(FULLY_CONNECTED, SIGMOID, 30);
    layers[1] = new LayerDescriptor(FULLY_CONNECTED, SIGMOID, 10);
    Network n(2, layers, 784);
    n.test(m, validation);
    input.close();
    required_output.close();
    /*int d[5][5] = {{17,24,1,8,15},
                  {23,5,7,14,16},
                  {4,6,13,20,22},
                  {10,12,19,21,3},
                  {11,18,25,2,9}};
    int w[3][3] = {{1,3,1},
                   {0,5,0},
                   {2,1,2}};
    Matrice input(5,5);
    for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < 5; j++)
                {
                    input.data[i][j] = d[i][j];
                }
        }
    Matrice weight(3,3);
    for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
                {
                    weight.data[i][j] = w[i][j];
                }
        }
    print_mtx(input);
    print_mtx(weight);
    Matrice convolved(3,3);
    print_mtx(convolved);
    weight = weight.rot180();
    convolution(input,weight,convolved,1);
    print_mtx(convolved);
    Matrice padded;
    padded = convolved.zero_padd(2,1,3,1);
    print_mtx(padded);*/
    return 0;
}
