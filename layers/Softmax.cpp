#include "layers.h"
#include <math.h>


Softmax::Softmax(int row, int col): FullyConnected(row, col, -1)
{
    this->layer_type = SOFTMAX;
    delete this->output_derivative[0];
    this->output_derivative[0] = new Matrix(this->outputlen, this->outputlen);
}

Softmax::~Softmax()
{
    ;
}

inline Matrix** Softmax::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **next_layers_error)
{
    cerr << "Softamx layer can only be an output layer!!!\n";
    throw exception();
}

inline void Softmax::layers_output(Matrix **input)
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
        this->output[0]->data[i][0] = output_helper.data[i][0] / nominator;
    }
}

inline Matrix** Softmax::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type)
{
    switch(costfunction_type)
    {
    case QUADRATIC_CF:
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error_helper[0][0].data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
        }
        this->derivate_layers_output(input);
        this->output_error[0][0] = this->output_derivative[0][0] * this->output_error_helper[0][0];
        return this->output_error;
    case LOG_LIKELIHOOD_CF:
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error[0][0].data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
        }
        return this->output_error;
    case CROSS_ENTROPY_CF:
        this->derivate_layers_output(input);
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error[0][0].data[i][0] = (this->output_derivative[0]->data[i][0] * (this->output[0][0].data[i][0] - required_output.data[i][0])) /
                                    (this->output[0][0].data[i][0] * (1 - this->output[0][0].data[i][0]));
        }
        return this->output_error;
    default:
        cerr << "Unknown cost function\n";
        throw exception();
    };
}

inline Matrix** Softmax::derivate_layers_output(Matrix **input)
{
    this->layers_output(input);
    for(int row = 0; row < this->outputlen; row ++)
    {
        for(int col = 0; col < this->outputlen; col++)
        {
            if(row == col)
            {
                this->output_derivative[0]->data[row][col] = this->output[0]->data[row][0] * (1 - this->output[0]->data[col][0]);
            }
            else
            {
                this->output_derivative[0]->data[row][col] = -1 * this->output[0]->data[row][0] * this->output[0]->data[col][0];
            }
        }
    }
    return this->output_derivative;
}



