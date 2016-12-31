#include "layers.h"

FullyConnected::FullyConnected(int row, int col, int prev_row, int neuron_type):
    output(row, col), weights(row, prev_row), biases(row, col), neuron(neuron_type)
{
    this->neuron_count = this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
    this->initialize_biases();
    this->initialize_weights();
}

FullyConnected::~FullyConnected()
{
    ;
}

inline void FullyConnected::layers_output(Matrice &input)
{
    //for(int i = 0; i < inputlen; i++)
        //cout<<input[i][0];
    for(int i = 0; i < this->outputlen; i++)
        {
            this->output.data[i][0] = this->neuron.neuron(this->weights.data[i], input.data, this->biases.data[i][0], input.get_row());
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
            mtx.data[i][0] = this->neuron.neuron_derivate(this->weights.data[i], input.data, this->biases.data[i][0], input.get_row());
        }
    return mtx;
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Matrice *weights, Matrice *biases)
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

void FullyConnected::set_input(double **input)
{
    ;
}

void FullyConnected::initialize_biases()
{
    ;
}

void FullyConnected::initialize_weights()
{
    ;
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

inline Matrice& FullyConnected::get_output()
{
    return this->output;
}

inline Matrice& FullyConnected::get_weights()
{
    return this->weights;
}

inline short FullyConnected::get_layer_type()
{
    return this->layer_type;
}

inline int FullyConnected::get_outputlen()
{
    return this->outputlen;
}

inline int FullyConnected::get_neuron_count()
{
    return this->neuron_count;
}
