#include "layers.h"

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type):
    neuron(neuron_type)
{
    this->output = new Matrice*[1];
    this->output[0] = new Matrice(row, 1);
    this->neuron_count = this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
    this->fmap = new Feature_map* [1];
    this->fmap[0] = new Feature_map(row, prev_row, 1, row);
}

FullyConnected::~FullyConnected()
{
    delete this->output[0];
    delete[] this->output;
    delete this->fmap[0];
    delete[] this->fmap;
}

inline void FullyConnected::layers_output(Matrice **input)
{
    Matrice inputparam(this->fmap[0]->biases[0][0].get_row(), this->fmap[0]->biases[0][0].get_col());
    inputparam += (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    this->output[0][0] = this->neuron.neuron(inputparam);
}

inline Matrice FullyConnected::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
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
            delta = hadamart_product(mtx, **output_derivate);
            delete output_derivate[0];
            delete[] output_derivate;
            return delta;
        case CROSS_ENTROPY_CF:
            switch(this->neuron_type)
                {
                case SIGMOID:
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            mtx.data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
                        }
                    return mtx;
                default:
                    output_derivate = this->derivate_layers_output(input);
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            delta.data[i][0] = (output_derivate[0]->data[i][0] * (this->output[0][0].data[i][0] - required_output.data[i][0])) /
                                                    (this->output[0][0].data[i][0] * (1 - this->output[0][0].data[i][0]));
                        }
                    delete output_derivate[0];
                    delete[] output_derivate;
                    return delta;
                }
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline Matrice** FullyConnected::derivate_layers_output(Matrice **input)
{
    Matrice **mtx;
    mtx = new Matrice* [1];
    mtx[0] = new Matrice;
    Matrice inputparam;
    inputparam = (this->fmap[0]->weights[0][0] * input[0][0] + this->fmap[0]->biases[0][0]);
    mtx[0][0] = this->neuron.neuron_derivate(inputparam);
    return mtx;
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

inline void FullyConnected::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void FullyConnected::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void FullyConnected::set_input(Matrice **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


inline Matrice** FullyConnected::backpropagate(Matrice **input, Feature_map** next_layers_fmaps, Feature_map** nabla,
                                          Matrice **delta, int next_layers_fmapcount)
{
    ///TODO think through this function from mathematical perspective!!!
    if(next_layers_fmapcount != 1)
        {
            cerr << "Currently the fully connected layer can be followed only by fully connected layers!\n";
            throw exception();
        }
    Matrice multiplied, **output_derivate;
    output_derivate = this->derivate_layers_output(input);
    multiplied = (next_layers_fmaps[0][0].weights[0]->transpose()) * delta[0][0];
    ///TODO maybe it would be necessary to reallocate the delta here, currently I do not think it'd be necessary
    delta[0][0] = hadamart_product(multiplied, **output_derivate);
    nabla[0][0].biases[0][0] = delta[0][0];
    nabla[0][0].weights[0][0] = delta[0][0] * input[0][0].transpose();
    delete output_derivate[0];
    delete[] output_derivate;
    return delta;
}

inline Matrice** FullyConnected::get_output()
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

void FullyConnected::set_weights(Matrice *w)
{
    this->fmap[0]->weights[0][0] = *w;
}

void FullyConnected::set_biases(Matrice *b)
{
    this->fmap[0]->biases[0][0] = *b;
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

void FullyConnected::store(std::ofstream &params)
{
    this->fmap[0]->store(params);
}

void FullyConnected::load(std::ifstream &params)
{
    this->fmap[0]->load(params);
}
