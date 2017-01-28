#include "layers.h"

FullyConnected::FullyConnected(int row, int prev_row, int neuron_type):
    output(row, 1), fmap(row, prev_row, 1), neuron(neuron_type)
{
    this->neuron_count = this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
}

FullyConnected::~FullyConnected()
{
    ;
}

inline void FullyConnected::layers_output(Matrice &input)
{
    for(int i = 0; i < this->outputlen; i++)
        {
            this->output.data[i][0] = this->neuron.neuron(this->fmap.weights[0][0].data[i], input.data, this->fmap.biases[0][0].data[i][0], input.get_row());
        }
}

inline Matrice FullyConnected::get_output_error(Matrice &input, double **required_output, int costfunction_type)
{
    Matrice mtx(this->outputlen, 1);
    Matrice delta(this->outputlen, 1);
    Matrice output_derivate(this->outputlen, 1);
    switch(costfunction_type)
        {
        case QUADRATIC_CF:
            for(int i = 0; i < this->outputlen; i++)
                {
                    mtx.data[i][0] = this->output.data[i][0] - required_output[i][0];
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
                            mtx.data[i][0] = this->output.data[i][0] - required_output[i][0];
                        }
                    return mtx;
                default:
                    output_derivate = this->derivate_layers_output(input);
                    for(int i = 0; i < this->outputlen; i++)
                        {
                            delta.data[i][0] = (output_derivate.data[i][0] * (this->output.data[i][0] - required_output[i][0])) /
                                                    (this->output.data[i][0] * (1 - this->output.data[i][0]));
                        }
                    return delta;
                }
        default:
            cerr << "Unknown cost function\n";
            throw exception();
        };
}

inline Matrice FullyConnected::derivate_layers_output(Matrice &input)
{
    Matrice mtx(this->outputlen, 1);
    for(int i = 0; i < this->outputlen; i++)
        {
            mtx.data[i][0] = this->neuron.neuron_derivate(this->fmap.weights[0][0].data[i], input.data, this->fmap.biases[0][0].data[i][0], input.get_row());
        }
    return mtx;
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, int prev_outputlen, Matrice *weights, Matrice *biases)
{
    for(int j = 0; j < this->outputlen; j++)
        {
            this->fmap.biases[0][0].data[j][0] -= learning_rate * biases->data[j][0];
            for(int k = 0; k < prev_outputlen; k++)
                {
                    this->fmap.weights[0][0].data[j][k] = regularization_rate * this->fmap.weights[0][0].data[j][k] - learning_rate * weights->data[j][k];
                }
        }
}

inline void FullyConnected::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    /*
    ///TAKE CARE!!! if this->dropout == false the function must return immediatelly!!!
    if((this->total_layers_num <= 2) || (this->dropout == false))
        return;
    ifstream rand;
    rand.open("/dev/urandom", ios::in);
    layers_bckup[0] = new int [this->total_layers_num];
    this->layers -= 1;
    for(int i = 0; i < this->total_layers_num; i++)
        layers_bckup[0][i] = this->layers[i];
    for(int i = 1; i < this->layers_num; i++)
        this->layers[i] >>= 1;
    this->layers += 1;
    layers_bckup[0] += 1;
    w_bckup[0] = new Matrice* [this->total_layers_num - 1];
    b_bckup[0] = new Matrice* [this->total_layers_num - 1];
    for(int i = 0; i < this->layers_num; i++)
        {
            w_bckup[0][i] = this->weights[i];
            b_bckup[0][i] = this->biases[i];
            this->biases[i] = new Matrice(this->layers[i], 1);
            this->weights[i] = new Matrice(this->layers[i], this->layers[i - 1]);
        }
    indexes[0] = new int* [this->total_layers_num - 2];
    int *tmp;
    for(int i = 0; i < this->total_layers_num - 2; i++)
        {
            indexes[0][i] = new int [this->layers[i]];
            tmp = new int[layers_bckup[0][i]];
            for(int j = 0; j < layers_bckup[0][i]; j++)
                tmp[j] = j;
            shuffle(tmp, layers_bckup[0][i], rand);
            for(int j = 0; j < this->layers[i]; j++)
                {
                    indexes[0][i][j] = tmp[j];
                }
            quickSort(indexes[0][i], 0, this->layers[i] - 1);
            delete[] tmp;
        }
    for(int j = 0; j < this->layers[0]; j++)
        {
            this->biases[0]->data[j][0] = b_bckup[0][0]->data[indexes[0][0][j]][0];
            for(int k = 0; k < this->layers[-1]; k++)
                {
                    this->weights[0]->data[j][k] = w_bckup[0][0]->data[indexes[0][0][j]][k];
                }
        }
    for(int i = 1; i < this->layers_num - 1; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    this->biases[i]->data[j][0] = b_bckup[0][i]->data[indexes[0][i][j]][0];
                    for(int k = 0; k < this->layers[i - 1]; k++)
                        {
                            this->weights[i]->data[j][k] = w_bckup[0][i]->data[indexes[0][i][j]][indexes[0][i - 1][k]];
                        }
                }
        }
    for(int j = 0; j < this->layers[this->layers_num - 1]; j++)
        {
            this->biases[this->layers_num - 1]->data[j][0] = b_bckup[0][this->layers_num - 1]->data[j][0];
            for(int k = 0; k < this->layers[this->layers_num - 2]; k++)
                {
                    this->weights[this->layers_num - 1]->data[j][k] = w_bckup[0][this->layers_num - 1]->data[j][indexes[0][this->layers_num - 2][k]];
                }
        }
    rand.close();*/
}

inline void FullyConnected::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    /*
    ///TAKE CARE!!! if this->dropout == false the function must return immediatelly!!!
    if((this->total_layers_num <= 2) || (this->dropout == false))
        return;
    for(int j = 0; j < this->layers[0]; j++)
        {
            b_bckup[0]->data[indexes[0][j]][0] = this->biases[0]->data[j][0];
            for(int k = 0; k < this->layers[-1]; k++)
                {
                    w_bckup[0]->data[indexes[0][j]][k] = this->weights[0]->data[j][k];
                }
        }
    for(int i = 1; i < this->layers_num - 1; i++)
        {
            for(int j = 0; j < this->layers[i]; j++)
                {
                    b_bckup[i]->data[indexes[i][j]][0] = this->biases[i]->data[j][0];
                    for(int k = 0; k < this->layers[i - 1]; k++)
                        {
                            w_bckup[i]->data[indexes[i][j]][indexes[i][k]] = this->weights[i]->data[j][k];
                        }
                }
        }
    for(int j = 0; j < this->layers[this->layers_num - 1]; j++)
        {
            b_bckup[this->layers_num - 1]->data[j][0] = this->biases[this->layers_num - 1]->data[j][0];
            for(int k = 0; k < this->layers[this->layers_num - 2]; k++)
                {
                    w_bckup[this->layers_num - 1]->data[j][indexes[this->layers_num - 2][k]] = this->weights[this->layers_num - 1]->data[j][k];
                }
        }
    for(int i = 0; i < this->layers_num; i++)
        {
            delete this->biases[i];
            this->biases[i] = b_bckup[i];
            delete this->weights[i];
            this->weights[i] = w_bckup[i];
            if(i < (this->layers_num - 1))
                delete[] indexes[i];
        }
    delete[] indexes;
    this->layers -= 1;
    layers_bckup -= 1;
    for(int i = 0; i < this->total_layers_num; i++)
        this->layers[i] = layers_bckup[i];
    this->layers += 1;
    delete[] layers_bckup;
*/
}

void FullyConnected::set_input(double **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


inline void FullyConnected::backpropagate(Matrice &input, Matrice& next_layers_weights, Matrice *nabla_b, Matrice *nabla_w, Matrice &delta)
{
    Matrice multiplied, output_derivate;
    output_derivate = this->derivate_layers_output(input);
    multiplied = next_layers_weights.transpose() * delta;
    delta = hadamart_product(multiplied, output_derivate);
    *(nabla_b) = delta;
    *(nabla_w) = delta * input.transpose();
}

inline Matrice* FullyConnected::get_output()
{
    return &(this->output);
}

inline Matrice* FullyConnected::get_weights()
{
    return this->fmap.weights[0];
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
