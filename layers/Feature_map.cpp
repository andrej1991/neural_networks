#include "layers.h"
#include "../opencl_setup.h"

Feature_map::Feature_map(int row, int col, int depth, int mapcount, int biascnt, OpenclSetup *env, bool initialization_needed):
                mapdepth(depth), row(row), col(col), openclenv(env), mapcount(mapcount)
{
    this->weights = new MatrixData* [depth];
    this->biases = new MatrixData* [depth];
    int biascount;
    if(biascnt > 1)
        biascount = biascnt;
    else
        biascount = 1;
    this->weights[0] = new MatrixData(mapcount * depth * row, col);
    this->biases[0] = new MatrixData(biascount, 1);
    if(env != NULL)
    {
        this->mtxop = new MatrixOperations(&(env->context), env->deviceIds);
        if(initialization_needed)
        {
            this->initialize_weights(&(env->context));
            this->initialize_biases(&(env->context));
        }
        else
        {
                this->biases[0][0].copy_to_opencl_buffer(&(env->context), &(this->mtxop[0].command_queue));
                this->weights[0][0].copy_to_opencl_buffer(&(env->context), &(this->mtxop[0].command_queue));
        }
    }
    else
        this->mtxop = NULL;
}

Feature_map::~Feature_map()
{

    delete this->weights[0];
    delete this->biases[0];
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
        for(int j = 0; j < this->biases[0][0].get_row(); j++)
        {
            random.read((char*)(&val), 2);
            (this->biases[0][0])[j][0] = (float)val/65000;
        }
        this->biases[0][0].copy_to_opencl_buffer(context, &(this->mtxop[0].command_queue));
    random.close();
}

void Feature_map::initialize_weights(cl_context *context)
{
    std::ifstream random;
    random.open("/dev/urandom", std::ios::in);
    short int val;

        for(int j = 0; j < this->weights[0][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[0][0].get_col(); k++)
            {
                random.read((char*)(&val), 2);
                (this->weights[0][0])[j][k] = (float)val/65000;
            }
        }
        this->weights[0][0].copy_to_opencl_buffer(context, &(this->mtxop[0].command_queue));
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
    {
        int s = this->weights[i][0].get_row() * this->weights[i][0].get_col() * sizeof(float);
        clEnqueueReadBuffer(this->mtxop[0].command_queue, this->weights[i][0].cl_mem_obj, CL_TRUE, 0, s, this->weights[i][0].data, 0, NULL, NULL);
        for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
            {
                params.write(reinterpret_cast<char *>(&((this->weights[i][0])[j][k])), sizeof(float));
            }
        }
    }
    for(int i = 0; i < this->mapdepth; i++)
    {
        int s = this->biases[i][0].get_row() * sizeof(float);
        clEnqueueReadBuffer(this->mtxop[0].command_queue, this->biases[i][0].cl_mem_obj, CL_TRUE, 0, s, this->biases[i][0].data, 0, NULL, NULL);
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
        {
            params.write(reinterpret_cast<char *>(&((this->biases[i][0])[j][0])), sizeof(float));
        }
    }
}

void Feature_map::load(std::ifstream &params)
{
    for(int i = 0; i < this->mapdepth; i++)
    {
        for(int j = 0; j < this->weights[i][0].get_row(); j++)
        {
            for(int k = 0; k < this->weights[i][0].get_col(); k++)
            {
                params.read(reinterpret_cast<char *>(&((this->weights[i][0])[j][k])), sizeof(float));
            }
        }
        this->weights[i][0].copy_to_opencl_buffer(&(this->openclenv->context), &(this->mtxop[0].command_queue));
    }
    for(int i = 0; i < this->mapdepth; i++)
    {
        for(int j = 0; j < this->biases[i][0].get_row(); j++)
        {
            params.read(reinterpret_cast<char *>(&((this->biases[i][0])[j][0])), sizeof(float));
        }
        this->biases[i][0].copy_to_opencl_buffer(&(this->openclenv->context), &(this->mtxop[0].command_queue));
    }
}
