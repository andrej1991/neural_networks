#include "layers.h"


Softmax::Softmax(int row, int col): FullyConnected(row, col, -1)
{
    this->layer_type = SOFTMAX;
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
    Matrice inputparam(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    inputparam += (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);

}

inline Matrice Softmax::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    cerr << "not implemented yet!";
    throw exception();
}

inline Matrice** Softmax::derivate_layers_output(Matrice **input)
{
    Matrice **mtx;
    mtx = new Matrice* [1];
    mtx[0] = new Matrice;
    Matrice inputparam;
    inputparam = (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
}


///these will be probably inherited from FullyConnected layer
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
