#include <random>
#include "layers.h"
#include "../matrix/matrix.h"

Feature_map::Feature_map(int row, int col, int depth ,int biasrow, int bias_col):
                mapdepth(depth)/*, row(row), col(col)*/
{
    this->weights = new Matrix* [depth];
    this->biases = new Matrix* [1];
    if(biasrow < 1 || bias_col < 1)
    {
        std::cerr << "Invalid row or colum in the Feature_map! row: " << biasrow << " col: " << bias_col << endl;
        throw std::invalid_argument("Invalid row or colum in the Feature_map!");
    }
    this->biases[0] = new Matrix(biasrow, bias_col);
    for(int i = 0; i < depth; i++)
    {
        this->weights[i] = new Matrix(row, col);
        //this->biases[i] = new Matrix(biasrow, bias_col);
    }
}

Feature_map::~Feature_map()
{
    for(int i = 0; i < this->mapdepth; i++)
    {
        delete this->weights[i];
    }
    delete this->biases[0];
    delete[] this->weights;
    delete[] this->biases;
}

void Feature_map::initialize_biases(double standard_deviation, double mean)
{
    std::random_device rand;
    std::normal_distribution<double> distribution (mean, standard_deviation);
    int i = 0;
    //for(int i = 0; i < this->mapdepth; i++)
    //{
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->biases[i][0].get_col(); k++)
            {
                this->biases[i][0].data[j][k] = distribution(rand);
            }
        }
    //}
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
        params.write(reinterpret_cast<char *>(this->biases[i][0].dv), sizeof(double)*this->biases[i][0].get_row()*this->biases[i][0].get_col());
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
        params.read(reinterpret_cast<char *>(this->biases[i][0].dv), sizeof(double)*this->biases[i][0].get_row()*this->biases[i][0].get_col());
    }

}
