#include "layers.h"
#include <math.h>


//Softmax::Softmax(int row, int col): Softmax(row, col, -1)
Softmax::Softmax(int row, int col)
{
    this->output = new Matrice*[1];
    this->output[0] = new Matrice(row, 1);
    this->neuron_count = this->outputlen = row;
    this->layer_type = SOFTMAX;
    this->fmap = new Feature_map* [1];
    this->fmap[0] = new Feature_map(row, col, 1, row);
}

Softmax::~Softmax()
{
    delete this->output[0];
    delete[] this->output;
    delete this->fmap[0];
    delete[] this->fmap;
}

inline Matrice** Softmax::backpropagate(Matrice **input, Feature_map** next_layers_fmaps, Feature_map** nabla, Matrice **next_layers_error, int next_layers_fmapcount)
{
    cerr << "Softamx layer can only be an output layer!!!\n";
    throw exception();
}

inline void Softmax::layers_output(Matrice **input)
{
    Matrice weighted_input(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    Matrice output_helper(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
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

inline Matrice Softmax::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    Matrice mtx(this->outputlen, 1);
    Matrice delta(this->outputlen, 1);
    Matrice **output_derivate;
    switch(costfunction_type)
        {
        case QUADRATIC_CF:
            for(int i = 0; i < this->outputlen; i++)
                {
                    mtx.data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
                }
            output_derivate = this->derivate_layers_output(input);
            delta = output_derivate[0][0] * mtx;
            delete output_derivate[0];
            delete[] output_derivate;
            return delta;
        case LOG_LIKELIHOOD:
            for(int i = 0; i < this->outputlen; i++)
                {
                    mtx.data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
                }
            return mtx;
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline Matrice** Softmax::derivate_layers_output(Matrice **input)
{
    Matrice **mtx;
    mtx = new Matrice* [1];
    mtx[0] = new Matrice(this->outputlen, this->outputlen);
    this->layers_output(input);
    for(int row = 0; row < this->outputlen; row ++)
        {
            for(int col = 0; col < this->outputlen; col++)
                {
                    if(row == col)
                        {
                            mtx[0]->data[row][col] = this->output[0]->data[row][0] * (1 - this->output[0]->data[col][0]);
                        }
                    else
                        {
                            mtx[0]->data[row][col] = -1 * this->output[0]->data[row][0] * this->output[0]->data[col][0];
                        }
                }
        }
    return mtx;
}


///these will be probably inherited from Softmax layer
/*void Softmax::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    ;
}

inline void Softmax::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void Softmax::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void Softmax::set_input(Matrice **input)
{
    ;
}

inline Matrice** Softmax::get_output()
{
    ;
}

inline Feature_map** Softmax::get_feature_maps()
{
    ;
}

inline short Softmax::get_layer_type()
{
    ;
}

inline int Softmax::get_output_len()
{
    ;
}

inline int Softmax::get_output_row()
{
    ;
}

inline int Softmax::get_output_col()
{
    ;
}

void Softmax::set_weights(Matrice *w)
{
    ;
}

void Softmax::set_biases(Matrice *b)
{
    ;
}

int Softmax::get_mapcount()
{
    ;
}

int Softmax::get_mapdepth()
{
    ;
}

int Softmax::get_weights_row()
{
    ;
}

int Softmax::get_weights_col()
{
    ;
}*/
void Softmax::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
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

inline void Softmax::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void Softmax::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void Softmax::set_input(Matrice **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


inline Matrice** Softmax::get_output()
{
    return this->output;
}

inline Feature_map** Softmax::get_feature_maps()
{
    return this->fmap;
}

inline short Softmax::get_layer_type()
{
    return this->layer_type;
}

inline int Softmax::get_output_row()
{
    return this->outputlen;
}

inline int Softmax::get_output_len()
{
    return this->outputlen;
}

inline int Softmax::get_output_col()
{
    return 1;
}

void Softmax::set_weights(Matrice *w)
{
    this->fmap[0]->weights[0][0] = *w;
}

void Softmax::set_biases(Matrice *b)
{
    this->fmap[0]->biases[0][0] = *b;
}

int Softmax::get_mapcount()
{
    return 1;
}

int Softmax::get_mapdepth()
{
    return 1;
}

int Softmax::get_weights_row()
{
    return this->fmap[0]->get_row();
}

int Softmax::get_weights_col()
{
    return this->fmap[0]->get_col();
}
