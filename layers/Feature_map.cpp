#include "layers.h"
#include "../opencl_setup.h"

Feature_map::Feature_map(int row, int col, int depth, int biascnt, OpenclSetup *env):
                mapdepth(depth), row(row), col(col)
{
    this->weights = new MatrixData* [depth];
    this->biases = new MatrixData* [depth];
    int biascount;
    if(biascnt > 1)
        biascount = biascnt;
    else
        biascount = 1;
    for(int i = 0; i < depth; i++)
    {
        this->weights[i] = new MatrixData(row, col);
        this->biases[i] = new MatrixData(biascount, 1);
    }
    this->initialize_weights(&(env->context));
    this->initialize_biases(&(env->context));
    if(env != NULL)
    {
        this->mtxop = new MatrixOperations(&(env->context), env->deviceIds);
    }
    else
        this->mtxop = NULL;
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
    if(this->mtxop != NULL)
        delete this->mtxop;
}

void Feature_map::initialize_biases(cl_context *context)
{
    std::ifstream random;
    random.open("/dev/urandom", std::ios::in);
    short int val;
    for(int i = 0; i < this->mapdepth; i++)
    {
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
        {
            random.read((char*)(&val), 2);
            (this->biases[i][0])[j][0] = val/65000;
        }
        this->biases[i][0].copy_to_opencl_buffer(context);
    }
    random.close();
}

void Feature_map::initialize_weights(cl_context *context)
{
    std::ifstream random;
    random.open("/dev/urandom", std::ios::in);
    short int val;
    for(int i = 0; i < this->mapdepth; i++)
    {
        for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
            {
                random.read((char*)(&val), 2);
                (this->weights[i][0])[j][k] = val/65000;
            }
        }
        this->weights[i][0].copy_to_opencl_buffer(context);
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
                    params.write(reinterpret_cast<char *>(&((this->weights[i][0])[j][k])), sizeof(double));
                }
        }
    for(int i = 0; i < this->mapdepth; i++)
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
            {
                params.write(reinterpret_cast<char *>(&((this->biases[i][0])[j][0])), sizeof(double));
            }
}

void Feature_map::load(std::ifstream &params)
{
    for(int i = 0; i < this->mapdepth; i++)
    for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
                {
                    params.read(reinterpret_cast<char *>(&((this->weights[i][0])[j][k])), sizeof(double));
                }
        }
    for(int i = 0; i < this->mapdepth; i++)
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
            {
                params.read(reinterpret_cast<char *>(&((this->biases[i][0])[j][0])), sizeof(double));
            }
}
