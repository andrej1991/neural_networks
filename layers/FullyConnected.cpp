#include <math.h>
#include <random>
#include "layers.h"

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type):
    neuron(neuron_type)
{
    this->output = new Matrix*[1];
    this->output[0] = new Matrix(row, 1);
    this->output_derivative = new Matrix*[1];
    this->output_derivative[0] = new Matrix(row, 1);
    this->output_error = new Matrix*[1];
    this->output_error[0] = new Matrix(row, 1);
    this->output_error_helper = new Matrix*[1];
    this->output_error_helper[0] = new Matrix(row, 1);
    this->layers_delta = new Matrix*[1];
    this->layers_delta[0] = new Matrix(row, 1);
    this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
    this->fmap = new Feature_map* [1];
    this->fmap[0] = new Feature_map(row, prev_row, 1, row);
    double deviation = sqrt(1/prev_row);
    this->fmap[0]->initialize_weights(deviation);
    this->fmap[0]->initialize_biases();
    this->dropout_happened = false;
    this->removed_rows = NULL;
    this->backup_outputlen = row;
}

FullyConnected::~FullyConnected()
{
    delete this->output[0];
    delete[] this->output;
    delete this->fmap[0];
    delete[] this->fmap;
    delete this->output_derivative[0];
    delete[] this->output_derivative;
    delete this->output_error[0];
    delete[] this->output_error;
    delete this->output_error_helper[0];
    delete[] this->output_error_helper;
    delete this->layers_delta[0];
    delete[] this->layers_delta;
}

inline void FullyConnected::layers_output(Matrix **input)
{
    Matrix inputparam(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    inputparam += (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    this->neuron.neuron(inputparam, this->output[0][0]);
}

inline Matrix** FullyConnected::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type)
{
    switch(costfunction_type)
        {
        case QUADRATIC_CF:
            for(int i = 0; i < this->outputlen; i++)
                {
                    this->output_error_helper[0][0].data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
                }
            this->derivate_layers_output(input);
            this->output_error[0][0] = hadamart_product(this->output_error_helper[0][0], this->output_derivative[0][0]);
            return this->output_error;
        case CROSS_ENTROPY_CF:
            switch(this->neuron_type)
                {
                case SIGMOID:
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            this->output_error[0][0].data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
                        }
                    return this->output_error;
                default:
                    this->derivate_layers_output(input);
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            this->output_error[0][0].data[i][0] = (this->output_derivative[0]->data[i][0] * (this->output[0][0].data[i][0] - required_output.data[i][0])) /
                                                    (this->output[0][0].data[i][0] * (1 - this->output[0][0].data[i][0]));
                        }
                    return this->output_error;
                }
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline Matrix** FullyConnected::derivate_layers_output(Matrix **input)
{
    //Matrix **mtx;
    //mtx = new Matrix* [1];
    //mtx[0] = new Matrix(this->outputlen, 1);
    Matrix inputparam;
    inputparam = (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    this->neuron.neuron_derivative(inputparam, this->output_derivative[0][0]);
    return this->output_derivative;
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    if((layer->get_fmap_count() != 1) + (layer->fmap[0]->get_mapdepth() != 1))
    {
        cerr << "the fully connected layer must have only one set of weights!!!\n";
        throw exception();
    }
    int prev_outputlen = this->fmap[0]->get_col();
    for(int j = 0; j < this->outputlen; j++)
    {
        this->fmap[0]->biases[0][0].data[j][0] -= learning_rate * layer->fmap[0]->biases[0]->data[j][0];
        for(int k = 0; k < prev_outputlen; k++)
        {
            this->fmap[0]->weights[0][0].data[j][k] = regularization_rate * this->fmap[0]->weights[0][0].data[j][k] - learning_rate * layer->fmap[0]->weights[0]->data[j][k];
        }
    }
}

void FullyConnected::set_input(Matrix **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


inline Matrix** FullyConnected::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix **delta)
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
    this->derivate_layers_output(input);
    multiplied = (next_layers_fmaps[0][0].weights[0]->transpose()) * delta[0][0];
    ///TODO maybe it would be necessary to reallocate the delta here, currently I do not think it'd be necessary
    this->layers_delta[0][0] = hadamart_product(multiplied, this->output_derivative[0][0]);
    nabla[0][0].biases[0][0] = this->layers_delta[0][0];
    nabla[0][0].weights[0][0] = this->layers_delta[0][0] * input[0][0].transpose();
    //delete output_derivate[0];
    //delete[] output_derivate;
    return this->layers_delta;
}

inline Matrix** FullyConnected::get_output()
{
    return this->output;
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
        this->backup_output = this->output[0];
        this->output[0] = new Matrix(remaining, 1);
        this->backup_output_derivative = this->output_derivative[0];
        this->output_derivative[0] = new Matrix(remaining, 1);
        this->backpup_output_error = this->output_error[0];
        this->output_error[0] = new Matrix(remaining, 1);
        if(probability == 0)
            throw exception();
        this->backup_output_error_helper = this->output_error_helper[0];
        this->output_error_helper[0] = new Matrix(remaining, 1);
        this->backup_layers_delta = this->layers_delta[0];
        this->layers_delta[0] = new Matrix(remaining, 1);
        this->outputlen = remaining;
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
        delete this->output[0];
        this->output[0] = this->backup_output;
        delete this->output_derivative[0];
        this->output_derivative[0] = this->backup_output_derivative;
        delete this->output_error[0];
        this->output_error[0] = this->backpup_output_error;
        delete this->output_error_helper[0];
        this->output_error_helper[0] = this->backup_output_error_helper;
        delete this->layers_delta[0];
        this->layers_delta[0] = this->backup_layers_delta;
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
