#include "layers.h"

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type):
    fmap(row, prev_row, 1, row), neuron(neuron_type)
{
    this->output = new Matrice*[1];
    this->output[0] = new Matrice(row, 1);
    this->neuron_count = this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
}

FullyConnected::~FullyConnected()
{
    delete this->output[0];
    delete[] this->output;
}

inline void FullyConnected::layers_output(Matrice **input)
{
    Matrice inputparam(this->fmap.biases[0][0].get_row(), this->fmap.biases[0][0].get_col());
    inputparam += (this->fmap.weights[0][0] * input[0][0] + this->fmap.biases[0][0]);
    this->output[0][0] = this->neuron.neuron(inputparam);
}

inline Matrice FullyConnected::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    Matrice mtx(this->outputlen, 1);
    Matrice delta(this->outputlen, 1);
    Matrice output_derivate(this->outputlen, 1);
    switch(costfunction_type)
        {
        case QUADRATIC_CF:
            for(int i = 0; i < this->outputlen; i++)
                {
                    mtx.data[i][0] = this->output[0][0].data[i][0] - required_output.data[i][0];
                }
            output_derivate = this->derivate_layers_output(input);
            delta = hadamart_product(mtx, output_derivate);
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
                            delta.data[i][0] = (output_derivate.data[i][0] * (this->output[0][0].data[i][0] - required_output.data[i][0])) /
                                                    (this->output[0][0].data[i][0] * (1 - this->output[0][0].data[i][0]));
                        }
                    return delta;
                }
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline Matrice FullyConnected::derivate_layers_output(Matrice **input)
{
    Matrice mtx(this->outputlen, 1);
    Matrice inputparam;
    inputparam = (this->fmap.weights[0][0] * input[0][0] + this->fmap.biases[0][0]);
    mtx = this->neuron.neuron_derivate(inputparam);
    return mtx;
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    if((layer->get_fmap_count() != 1) + (layer->fmap[0]->get_mapdepth() != 1))
        {
            cerr << "the fully connected layer must have only one set of weights!!!\n";
            throw exception();
        }
    int prev_outputlen = this->fmap.get_col();
    for(int j = 0; j < this->outputlen; j++)
        {
            this->fmap.biases[0][0].data[j][0] -= learning_rate * layer->fmap[0]->biases[0]->data[j][0];
            for(int k = 0; k < prev_outputlen; k++)
                {
                    this->fmap.weights[0][0].data[j][k] = regularization_rate * this->fmap.weights[0][0].data[j][k] - learning_rate * layer->fmap[0]->weights[0]->data[j][k];
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


inline void FullyConnected::backpropagate(Matrice **input, Feature_map** next_layers_fmaps, Feature_map** nabla,
                                          Matrice &delta, int next_layers_fmapcount)
//inline void FullyConnected::backpropagate(Matrice **input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &delta)
{
    ///TODO think through this function from mathematical perspective!!!
    if(next_layers_fmapcount != 1)
        {
            cerr << "Currently the fully connected layer can be followed only by fully connected layers!\n";
            throw exception();
        }
    Matrice multiplied, output_derivate;
    output_derivate = this->derivate_layers_output(input);
    //int debug1 = next_layers_fmaps[0][0].weights[0][0].get_row();
    //int debug2 = next_layers_fmaps[0][0].weights[0]->get_col();
    //int debug3 = delta.get_row();
    //int debug4 = delta.get_col();
    multiplied = (next_layers_fmaps[0][0].weights[0]->transpose()) * delta;
    //int debug5 = 0;
    delta = hadamart_product(multiplied, output_derivate);
    nabla[0][0].biases[0][0] = delta;
    nabla[0][0].weights[0][0] = delta * input[0][0].transpose();
}

inline Matrice** FullyConnected::get_output()
{
    return this->output;
}

inline Feature_map** FullyConnected::get_feature_maps()
{
    Feature_map *helper = &(this->fmap);
    Feature_map **h2 = &helper;
    //int debug1 = h2[0][0].weights[0][0].get_row();
    return h2;
}

inline short FullyConnected::get_layer_type()
{
    return this->layer_type;
}

inline int FullyConnected::get_outputlen()
{
    return this->outputlen;
}

void FullyConnected::set_weights(Matrice *w)
{
    this->fmap.weights[0][0] = *w;
}

void FullyConnected::set_biases(Matrice *b)
{
    this->fmap.biases[0][0] = *b;
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
    return this->fmap.get_row();
}

int FullyConnected::get_weights_col()
{
    return this->fmap.get_col();
}
