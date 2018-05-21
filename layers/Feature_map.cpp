#include "layers.h"

Feature_map::Feature_map(int row, int col, int depth, int biascnt, bool initializtion_needed):
                mapdepth(depth), row(row), col(col)
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
    if(initializtion_needed)
    {
        this->initialize_weights();
        this->initialize_biases();
    }
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

int Feature_map::get_col()
{
    return this->col;
}

int Feature_map::get_row()
{
    return this->row;
}

int Feature_map::get_mapdepth()
{
    return this->mapdepth;
}

void Feature_map::store(std::ofstream &params)
{
    for(int i = 0; i < this->mapdepth; i++)
    for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
                {
                    params.write(reinterpret_cast<char *>(&(this->weights[i][0].data[j][k])), sizeof(double));
                }
        }
    for(int i = 0; i < this->mapdepth; i++)
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
            {
                params.write(reinterpret_cast<char *>(&(this->biases[i][0].data[j][0])), sizeof(double));
            }
}

void Feature_map::load(std::ifstream &params)
{
    for(int i = 0; i < this->mapdepth; i++)
    for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
                {
                    params.read(reinterpret_cast<char *>(&(this->weights[i][0].data[j][k])), sizeof(double));
                }
        }
    for(int i = 0; i < this->mapdepth; i++)
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
            {
                params.read(reinterpret_cast<char *>(&(this->biases[i][0].data[j][0])), sizeof(double));
            }
}
