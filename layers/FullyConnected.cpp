#include "layers.h"

FullyConnected::FullyConnected()
{
    ;
}

FullyConnected::~FullyConnected()
{
    ;
}

inline void FullyConnected::layers_output(double **input, int layer)
{
    for(int i = 0; i < this->outputlen; i++)
        {
            this->output.data[i][0] = this->neuron.neuron(this->weights.data[i], input, this->biases.data[i][0], this->outputlen);
        }
}

inline Matrice FullyConnected::get_output_error(double **input, double **required_output, int inputlen, int costfunction_type)
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
            output_derivate = this->derivate_layers_output(input, inputlen);
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
                    output_derivate = this->derivate_layers_output(input, inputlen);
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

inline Matrice FullyConnected::derivate_layers_output(double **input, int inputlen)
{
    Matrice mtx(this->outputlen, 1);
    for(int i = 0; i < this->outputlen; i++)
        {
            mtx.data[i][0] = this->neuron.neuron_derivate(this->weights.data[i], input, this->biases.data[i][0], inputlen);
        }
    return mtx;
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice **weights, Matrice **biases)
{
    ;
}

inline void FullyConnected::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;
}

inline void FullyConnected::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;
}

void FullyConnected::set_input(double **input, int input_len)
{
    ;
}

