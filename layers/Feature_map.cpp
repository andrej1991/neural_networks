#include "layers.h"

Feature_map::Feature_map(int row, int col, int depth, int biascnt):
                mapdepth(depth)
{
    this->weights = new Matrice* [depth];
    this->biases = new Matrice* [depth];
    int biascount;
    if(biascnt > 1)
        biascount = biascnt;
    else
        biascount = 1;
    for(int i = 0; i < depth; i++)
        {
            this->weights[i] = new Matrice(row, col);
            this->biases[i] = new Matrice(biascount, 1);
        }
    this->initialize_weights();
    this->initialize_biases();
}

Feature_map::~Feature_map()
{
    for(int i = 0; i < this->mapdepth; i++)
        {
            delete this->weights[i];
            delete this->biases[i];
        }
    delete[] this->weights;
    delete[] this->biases;
}

void Feature_map::initialize_biases()
{
    ifstream random;
    random.open("/dev/urandom", ios::in);
    short int val;
    for(int i = 0; i < this->mapdepth; i++)
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
            {
                random.read((char*)(&val), 2);
                this->biases[i][0].data[j][0] = val;
                this->biases[i][0].data[j][0] /= 63000;
            }
    random.close();
}

void Feature_map::initialize_weights()
{
    ifstream random;
    random.open("/dev/urandom", ios::in);
    short int val;
    for(int i = 0; i < this->mapdepth; i++)
    for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
                {
                    random.read((char*)(&val), 2);
                    this->weights[i][0].data[j][k] = val;
                    this->weights[i][0].data[j][k] /= 63000;
                }
        }
    random.close();
}
