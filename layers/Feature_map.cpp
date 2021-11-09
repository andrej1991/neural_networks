#include <random>
#include "layers.h"
#include "../matrix/matrix.h"

Feature_map::Feature_map(int row, int col, int depth ,int biascnt):
                mapdepth(depth)/*, row(row), col(col)*/
{
    this->weights = new Matrix* [depth];
    this->biases = new Matrix* [depth];
    int biascount;
    if(biascnt > 1)
        biascount = biascnt;
    else
        biascount = 1;
    for(int i = 0; i < depth; i++)
        {
            this->weights[i] = new Matrix(row, col);
            this->biases[i] = new Matrix(biascount, 1);
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

void Feature_map::initialize_biases(double standard_deviation, double mean)
{
    std::random_device rand;
    std::normal_distribution<double> distribution (mean, standard_deviation);
    for(int i = 0; i < this->mapdepth; i++)
    {
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
        {
            this->biases[i][0].data[j][0] = distribution(rand);
        }
    }
}

void Feature_map::initialize_weights(double standard_deviation, double mean)
{
    std::random_device rand;
    std::normal_distribution<double> distribution (mean, standard_deviation);
    for(int i = 0; i < this->mapdepth; i++)
    {
        for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
            {
                this->weights[i][0].data[j][k] = distribution(rand);
            }
        }
    }
}

int Feature_map::get_col()
{
    return this->weights[0]->get_col();
}

int Feature_map::get_row()
{
    return this->weights[0]->get_row();
}

int Feature_map::get_mapdepth()
{
    return this->mapdepth;
}

void Feature_map::store(std::ofstream &params)
{
    for(int i = 0; i < this->mapdepth; i++)
    {
        params.write(reinterpret_cast<char *>(this->weights[i][0].dv), sizeof(double)*this->weights[i][0].get_row()*this->weights[i][0].get_col());
    }

    for(int i = 0; i < this->mapdepth; i++)
    {
        params.write(reinterpret_cast<char *>(this->biases[i][0].dv), sizeof(double)*this->biases[i][0].get_row());
    }

}

void Feature_map::load(std::ifstream &params)
{
    for(int i = 0; i < this->mapdepth; i++)
    {
        params.read(reinterpret_cast<char *>(this->weights[i][0].dv), sizeof(double)*this->weights[i][0].get_row()*this->weights[i][0].get_col());
    }


    for(int i = 0; i < this->mapdepth; i++)
    {
        params.read(reinterpret_cast<char *>(this->biases[i][0].dv), sizeof(double)*this->biases[i][0].get_row());
    }

}
