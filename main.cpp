#include <iostream>
#include <fstream>
#include "MNIST_data.h"
#include "network.h"
#include "matrice.h"

using namespace std;


int main()
{
    ifstream input, required_output;
    input.open("/home/andrej/hdd/dokumentumok/neural_networks/data/training_data/input.dat", ios::in|ios::binary);
    required_output.open("/home/andrej/hdd/dokumentumok/neural_networks/data/training_data/required_output.dat", ios::in|ios::binary);
    MNIST_data *m[50000];
    for(int i = 0; i < 50000; i++)
        {
            m[i] = new MNIST_data(784, 10);
            m[i]->load_data(input, required_output);
        }
    cout << "Hello world!" << endl;
    int layers[3] = {784, 30, 10};
    Network n(3, layers);
    n.test(m);
    input.close();
    required_output.close();
    return 0;
}
