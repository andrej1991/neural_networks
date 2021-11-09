#include <math.h>
#include <random>
#include "layers.h"

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type):
    neuron(neuron_type)
{
    this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
    this->fmap = new Feature_map* [1];
    this->fmap[0] = new Feature_map(row, prev_row, 1, row);
    double deviation = sqrt(1.0/prev_row);
    this->fmap[0]->initialize_weights(deviation);
    this->fmap[0]->initialize_biases();
    this->dropout_happened = false;
    this->removed_rows = NULL;
    this->backup_outputlen = row;
    this->threadcount = 1;
    this->build_dinamic_data();
}

FullyConnected::~FullyConnected()
{
    delete this->fmap[0];
    delete[] this->fmap;
    this->destroy_dinamic_data();
}

void FullyConnected::destroy_dinamic_data()
{
    for(int i = 0; i < this->threadcount; i++)
    {
        delete this->output[i][0];
        delete this->output_derivative[i][0];
        delete this->output_error[i][0];
        delete this->output_error_helper[i][0];
        delete this->layers_delta[i][0];
        delete[] this->output[i];
        delete[] this->output_derivative[i];
        delete[] this->output_error[i];
        delete[] this->output_error_helper[i];
        delete[] this->layers_delta[i];
    }
    delete[] this->output;
    delete[] this->output_derivative;
    delete[] this->output_error;
    delete[] this->output_error_helper;
    delete[] this->layers_delta;
}

void FullyConnected::build_dinamic_data()
{
    this->output = new Matrix**[this->threadcount];
    this->output_derivative = new Matrix**[this->threadcount];
    this->output_error = new Matrix**[this->threadcount];
    this->output_error_helper = new Matrix**[this->threadcount];
    this->layers_delta = new Matrix**[this->threadcount];
    for(int i = 0; i < this->threadcount; i++)
    {
        this->output[i] = new Matrix* [1];
        this->output_derivative[i] = new Matrix*[1];
        this->output_error[i] = new Matrix*[1];
        this->output_error_helper[i] = new Matrix*[1];
        this->layers_delta[i] = new Matrix*[1];
        this->output[i][0] = new Matrix(this->outputlen, 1);
        this->output_derivative[i][0] = new Matrix(this->outputlen, 1);
        this->output_error[i][0] = new Matrix(this->outputlen, 1);
        this->output_error_helper[i][0] = new Matrix(this->outputlen, 1);
        this->layers_delta[i][0] = new Matrix(this->outputlen, 1);
    }
}

void FullyConnected::layers_output(Matrix **input, int threadindex)
{
    Matrix inputparam(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    inputparam += (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    this->neuron.neuron(inputparam, this->output[threadindex][0][0]);
}

Matrix** FullyConnected::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
{
    switch(costfunction_type)
    {
    case QUADRATIC_CF:
        for(int i = 0; i < this->outputlen; i++)
        {
            this->output_error_helper[threadindex][0][0].data[i][0] = this->output[threadindex][0][0].data[i][0] - required_output.data[i][0];
        }
        this->derivate_layers_output(input, threadindex);
        this->output_error[threadindex][0][0] = hadamart_product(this->output_error_helper[threadindex][0][0], this->output_derivative[threadindex][0][0]);
        return this->output_error[threadindex];
    case CROSS_ENTROPY_CF:
        switch(this->neuron_type)
        {
        case SIGMOID:
            for(int i = 0; i < this->outputlen; i++)
            {
                this->output_error[threadindex][0][0].data[i][0] = this->output[threadindex][0][0].data[i][0] - required_output.data[i][0];
            }
            return this->output_error[threadindex];
        default:
            this->derivate_layers_output(input, threadindex);
            for(int i = 0; i < this->outputlen; i++)
            {
                this->output_error[threadindex][0][0].data[i][0] = (this->output_derivative[threadindex][0]->data[i][0] * (this->output[threadindex][0][0].data[i][0] - required_output.data[i][0])) /
                                                       (this->output[threadindex][0][0].data[i][0] * (1 - this->output[threadindex][0][0].data[i][0]));
            }
            return this->output_error[threadindex];
        }
    default:
        cerr << "Unknown cost function\n";
        throw exception();
    };
}

Matrix** FullyConnected::derivate_layers_output(Matrix **input, int threadindex)
{
    Matrix inputparam;
    inputparam = (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    this->neuron.neuron_derivative(inputparam, this->output_derivative[threadindex][0][0]);
    return this->output_derivative[threadindex];
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *gradient)
{
    if((gradient->get_fmap_count() != 1) + (gradient->fmap[0]->get_mapdepth() != 1))
    {
        cerr << "the fully connected layer must have only one set of weights!!!\n";
        throw exception();
    }
    int prev_outputlen = this->fmap[0]->get_col();
    for(int j = 0; j < this->outputlen; j++)
    {
        this->fmap[0]->biases[0][0].data[j][0] -= learning_rate * gradient->fmap[0]->biases[0]->data[j][0];
        for(int k = 0; k < prev_outputlen; k++)
        {
            this->fmap[0]->weights[0][0].data[j][k] = regularization_rate * this->fmap[0]->weights[0][0].data[j][k] - learning_rate * gradient->fmap[0]->weights[0]->data[j][k];
        }
    }
}

void FullyConnected::set_input(Matrix **input, int threadindex)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


Matrix** FullyConnected::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **delta, int threadindex)
{
    ///TODO think through this function from mathematical perspective!!!
    Feature_map** next_layers_fmaps = next_layer->get_feature_maps();
    int next_layers_fmapcount = next_layer->get_mapcount();
    if(next_layers_fmapcount != 1)
        {
            cerr << "Currently the fully connected layer can be followed only by fully connected layers!\n";
            throw exception();
        }
    Matrix multiplied;
    this->derivate_layers_output(input, threadindex);
    multiplied = (next_layers_fmaps[0][0].weights[0]->transpose()) * delta[0][0];
    ///TODO maybe it would be necessary to reallocate the delta here, currently I do not think it'd be necessary
    this->layers_delta[threadindex][0][0] = hadamart_product(multiplied, this->output_derivative[threadindex][0][0]);
    nabla[0][0].biases[0][0] = this->layers_delta[threadindex][0][0];
    nabla[0][0].weights[0][0] = this->layers_delta[threadindex][0][0] * input[0][0].transpose();
    return this->layers_delta[threadindex];
}

inline int FullyConnected::get_threadcount()
{
    return this->threadcount;
}

void FullyConnected::set_threadcount(int threadcnt)
{
    this->destroy_dinamic_data();
    this->threadcount = threadcnt;
    this->build_dinamic_data();
}

inline Matrix** FullyConnected::get_output(int threadindex)
{
    return this->output[threadindex];
}

inline Feature_map** FullyConnected::get_feature_maps()
{
    return this->fmap;
}

inline short FullyConnected::get_layer_type()
{
    return this->layer_type;
}

inline int FullyConnected::get_output_row()
{
    return this->outputlen;
}

inline int FullyConnected::get_output_len()
{
    return this->outputlen;
}

inline int FullyConnected::get_output_col()
{
    return 1;
}

int FullyConnected::get_mapcount()
{
    return 1;
}

int FullyConnected::get_mapdepth()
{
    return 1;
}

int FullyConnected::get_weights_row()
{
    return this->fmap[0]->get_row();
}

int FullyConnected::get_weights_col()
{
    return this->fmap[0]->get_col();
}

Matrix FullyConnected::drop_out_some_neurons(double probability, Matrix *colums_to_remove)
{
    if(this->removed_rows == NULL)
    {
        this->removed_rows = new Matrix(this->outputlen, 1);
    }
    this->removed_rows->zero();
    if(probability == 0 and colums_to_remove == NULL)
    {
        return this->removed_rows[0];
    }
    std::random_device rand;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    int remaining, dropped_out = 0;
    if(probability > 0.0)
    {
        for (int i = 0; i < this->outputlen; i++)
        {
            if(distribution(rand) < probability)
            {
                this->removed_rows->data[i][0] = 1;
                dropped_out++;
            }
            else
            {
                this->removed_rows->data[i][0] = 0;
            }
        }
    }
    if(dropped_out > 0)
    {
        this->dropout_happened = true;
    }
    else
    {
        this->dropout_happened = false;
        if(colums_to_remove == NULL)
            return this->removed_rows[0];
    }
    remaining = this->outputlen - dropped_out;
    this->backup_weights = this->fmap[0]->weights[0];
    if(dropout_happened)
    {
        this->backup_biases = this->fmap[0]->biases[0];
        this->fmap[0]->weights[0] = this->backup_weights->remove_rows(this->removed_rows[0]);
        this->fmap[0]->biases[0] = this->backup_biases->remove_rows(this->removed_rows[0]);
        this->backup_output = this->output;
        this->backup_output_derivative = this->output_derivative;
        this->backpup_output_error = this->output_error;
        if(probability == 0)
            throw exception();
        this->backup_output_error_helper = this->output_error_helper;
        this->backup_layers_delta = this->layers_delta;
        this->outputlen = remaining;
        this->build_dinamic_data();
    }
    if(colums_to_remove != NULL)
    {
        Matrix *backup = this->fmap[0]->weights[0];
        this->fmap[0]->weights[0] = backup->remove_colums(colums_to_remove[0]);
        if(dropout_happened)
            delete backup;
    }
    return this->removed_rows[0];
}

void FullyConnected::restore_neurons(Matrix *removed_colums)
{
    if(this->dropout_happened == false and removed_colums == NULL)
        return;
    int backup_col = this->backup_weights->get_col();
    int r = 0, c = 0;
    for(int i = 0; i < this->backup_outputlen; i++)
    {
        if(dropout_happened and this->removed_rows->data[i][0] != 1)
        {
            this->backup_biases->data[i][0] = this->fmap[0]->biases[0]->data[r][0];
        }
        for(int j = 0; j < backup_col; j++)
        {
            if(this->removed_rows->data[i][0] != 1 and (removed_colums == NULL or removed_colums->data[j][0] != 1))
            {
                this->backup_weights->data[i][j] = this->fmap[0]->weights[0]->data[r][c];
                c++;
            }
        }
        if(this->removed_rows->data[i][0] != 1)
            r++;
        c = 0;
    }
    delete this->fmap[0]->weights[0];
    this->fmap[0]->weights[0] = this->backup_weights;
    if(dropout_happened)
    {
        delete this->fmap[0]->biases[0];
        this->fmap[0]->biases[0] = this->backup_biases;
        this->destroy_dinamic_data();
        this->output = this->backup_output;
        this->output_derivative = this->backup_output_derivative;
        this->output_error = this->backpup_output_error;
        this->output_error_helper = this->backup_output_error_helper;
        this->layers_delta = this->backup_layers_delta;
        this->outputlen = this->backup_outputlen;
    }
}

void FullyConnected::store(std::ofstream &params)
{
    this->fmap[0]->store(params);
}

void FullyConnected::load(std::ifstream &params)
{
    this->fmap[0]->load(params);
}
