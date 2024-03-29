#include "layers.h"
#include <math.h>


Softmax::Softmax(int row, int col): FullyConnected(row, col, -1)
{
    this->layer_type = SOFTMAX;
    delete this->output_derivative[0][0];
    this->output_derivative[0][0] = new Matrix(this->outputlen, this->outputlen);
}

Softmax::~Softmax()
{
    ;
}

inline Matrix** Softmax::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error, int threadindex)
{
    cerr << "Softamx layer can only be an output layer!!!\n";
    throw exception();
}

void Softmax::layers_output(Matrix **input, int threadindex)
{
    Matrix weighted_input(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    Matrix output_helper(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    double nominator = 0;
    double helper;
    weighted_input += (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    for(int i = 0; i < this->outputlen; i++)
    {
        output_helper.data[i][0] = exp(weighted_input.data[i][0]);
        nominator += output_helper.data[i][0];
    }
    for(int i = 0; i < this->outputlen; i++)
    {
        this->output[threadindex][0]->data[i][0] = output_helper.data[i][0] / nominator;
    }
}

Matrix** Softmax::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    switch(costfunction_type)
    {
    case QUADRATIC_CF:
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error_helper[threadindex][0][0].data[i][0] = this->output[threadindex][0][0].data[i][0] - required_output.data[i][0];
        }
        this->derivate_layers_output(input, threadindex);
        this->output_error[threadindex][0][0] = this->output_derivative[threadindex][0][0] * this->output_error_helper[threadindex][0][0];
        return this->output_error[threadindex];
    case LOG_LIKELIHOOD_CF:
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error[threadindex][0][0].data[i][0] = this->output[threadindex][0][0].data[i][0] - required_output.data[i][0];
        }
        return this->output_error[threadindex];
    case CROSS_ENTROPY_CF:
        this->derivate_layers_output(input, threadindex);
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error[threadindex][0][0].data[i][0] = (this->output_derivative[threadindex][0]->data[i][0] * (this->output[threadindex][0][0].data[i][0] - required_output.data[i][0])) /
                                    (this->output[threadindex][0][0].data[i][0] * (1 - this->output[threadindex][0][0].data[i][0]));
        }
        return this->output_error[threadindex];
    default:
        cerr << "Unknown cost function\n";
        throw exception();
    };
}

Matrix** Softmax::derivate_layers_output(Matrix **input, int threadindex)
{
    this->layers_output(input, threadindex);
    for(int row = 0; row < this->outputlen; row ++)
    {
        for(int col = 0; col < this->outputlen; col++)
        {
            if(row == col)
            {
                this->output_derivative[threadindex][0]->data[row][col] = this->output[threadindex][0]->data[row][0] * (1 - this->output[threadindex][0]->data[col][0]);
            }
            else
            {
                this->output_derivative[threadindex][0]->data[row][col] = -1 * this->output[threadindex][0]->data[row][0] * this->output[threadindex][0]->data[col][0];
            }
        }
    }
    return this->output_derivative[threadindex];
}



