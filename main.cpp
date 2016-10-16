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
    /*Matrice m1(10, 1), m2(1, 10);
    Matrice m3 = m1 * m2;
    Matrice **m4, m5(10, 1);
    m4 = new Matrice* [1];
    m4[0] = new Matrice [2];
    for(int i = 0; i < 10; i++)
        m5.data[i][0] = i;
    m4[0][1] = m5;
    for(int i = 0; i < 10; i++)
        {
            cout << m4[0][1].data[i][0];
        }*/
    return 0;
}
