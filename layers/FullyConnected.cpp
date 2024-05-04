#include <math.h>
#include <random>
#include "layers.h"

FullyConnected::FullyConnected(int row, Layer **layers, vector<int> input_from, int neuron_type, int _my_index):
    neuron(neuron_type), my_index(_my_index)
{
    this->network_layers = layers;
    this->gets_input_from_ = input_from;
    this->outputlen = row;
    this->layer_type = FULLY_CONNECTED;
    double deviation = 1.0;
    double mean = 0.0;
    this->fmap = new Feature_map* [inputs.size()];
    for(int i = 0; i < input_from.size(); i++)
    {
        deviation = 1.0/sqrt(this->network_layers[input_from[i]]->get_output_len());
        this->fmap[i] = new Feature_map(row, this->network_layers[input_from[i]]->get_output_len(), 1, row);
        if(neuron_type == SIGMOID)
        {
            mean = 0.5;
        }
        else if(neuron_type == RELU || neuron_type == LEAKY_RELU)
        {
            mean = deviation;
        }
        this->fmap[i]->initialize_weights(deviation, mean);
        this->fmap[i]->initialize_biases();
    }
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
        delete this->activation_input[i][0];
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
    delete[] this->activation_input;
    delete[] this->output_derivative;
    delete[] this->output_error;
    delete[] this->output_error_helper;
    delete[] this->layers_delta;
}

void FullyConnected::build_dinamic_data()
{
    this->output = new Matrix**[this->threadcount];
    this->activation_input = new Matrix**[this->threadcount];
    this->output_derivative = new Matrix**[this->threadcount];
    this->output_error = new Matrix**[this->threadcount];
    this->output_error_helper = new Matrix**[this->threadcount];
    this->layers_delta = new Matrix**[this->threadcount];
    for(int i = 0; i < this->threadcount; i++)
    {
        this->output[i] = new Matrix* [1];
        this->activation_input[i] = new Matrix* [1];
        this->output_derivative[i] = new Matrix*[1];
        this->output_error[i] = new Matrix*[1];
        this->output_error_helper[i] = new Matrix*[1];
        this->layers_delta[i] = new Matrix*[1];
        this->output[i][0] = new Matrix(this->outputlen, 1);
        this->activation_input[i][0] = new Matrix(this->outputlen, 1);
        this->output_derivative[i][0] = new Matrix(this->outputlen, 1);
        this->output_error[i][0] = new Matrix(this->outputlen, 1);
        this->output_error_helper[i][0] = new Matrix(this->outputlen, 1);
        this->layers_delta[i][0] = new Matrix(this->outputlen, 1);
    }
}

void FullyConnected::set_layers_inputs(vector<Matrix***> inputs_)
{
    this->inputs.clear();
    for(Matrix ***inp : inputs_)
    {
        this->inputs.push_back(inp);
    }
}

void FullyConnected::layers_output(Matrix **input, int threadindex)
{
    if(input != NULL)
    {
        cerr << "something needs to be figured out for getting the output of standalone layers";
        throw exception();
    }
    this->activation_input[threadindex][0][0].zero();
    for(int i = 0; i < this->inputs.size(); i++)
    {
        cout << my_index << "  " << i << endl;
        //print_mtx(this->fmap[i]->weights[0][0]);
        print_mtx(this->inputs[i][threadindex][0][0]);
        print_mtx(this->fmap[0]->biases[0][0]);
        print_mtx(this->activation_input[threadindex][0][0]);
        weighted_output(this->fmap[i]->weights[0][0], this->inputs[i][threadindex][0][0], this->fmap[0]->biases[0][0], this->activation_input[threadindex][0][0]);
    }
    this->neuron.neuron(this->activation_input[threadindex][0][0], this->output[threadindex][0][0]);
}

Matrix* FullyConnected::get_output_error(Matrix **input, Matrix &required_output, int costfunction_type, int threadindex)
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
        return this->output_error[threadindex][0];
    case CROSS_ENTROPY_CF:
        switch(this->neuron_type)
        {
        case SIGMOID:
            for(int i = 0; i < this->outputlen; i++)
            {
                this->output_error[threadindex][0][0].data[i][0] = this->output[threadindex][0][0].data[i][0] - required_output.data[i][0];
            }
            return this->output_error[threadindex][0];
        default:
            this->derivate_layers_output(input, threadindex);
            for(int i = 0; i < this->outputlen; i++)
            {
                this->output_error[threadindex][0][0].data[i][0] = (this->output_derivative[threadindex][0]->data[i][0] * (this->output[threadindex][0][0].data[i][0] - required_output.data[i][0])) /
                                                       (this->output[threadindex][0][0].data[i][0] * (1 - this->output[threadindex][0][0].data[i][0]));
            }
            return this->output_error[threadindex][0];
        }
    default:
        cerr << "Unknown cost function\n";
        throw exception();
    };
}

Matrix** FullyConnected::derivate_layers_output(Matrix **input, int threadindex)
{
    this->neuron.neuron_derivative(this->activation_input[threadindex][0][0], this->output_derivative[threadindex][0][0]);
    return this->output_derivative[threadindex];
}

void FullyConnected::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *gradient)
{
    /*if((gradient->get_fmap_count() != 1) + (gradient->fmap[0]->get_mapdepth() != 1))
    {
        cerr << "the fully connected layer must have only one set of weights!!!\n";
        throw exception();
    }*/
    for(int i = 0; i < this->gets_input_from().size(); i++)
    {
        int prev_outputlen = this->fmap[i]->get_col();
        for(int j = 0; j < this->outputlen; j++)
        {
            for(int k = 0; k < prev_outputlen; k++)
            {
                this->fmap[i]->weights[0][0].data[j][k] = regularization_rate * this->fmap[i]->weights[0][0].data[j][k] - learning_rate * gradient->fmap[i]->weights[0]->data[j][k];
            }
        }
    }
    this->fmap[0]->biases[0][0] += gradient->fmap[0]->biases[0][0] * (-1 * learning_rate);
}

void FullyConnected::set_input(Matrix **input, int threadindex)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}


Matrix** FullyConnected::backpropagate(Matrix **input, Layer *next_layer, Feature_map** nabla, Matrix ***deltas, int threadindex)
{
    this->derivate_layers_output(NULL, threadindex);
    ///TODO think through this function from mathematical perspective!!!
    Feature_map** next_layers_fmaps;
    int next_layers_fmapcount, indexOfMyInput, j, l;
    vector<int> next_layers_inputs;
    this->layers_delta[threadindex][0][0].zero();
    for(int i : this->sends_output_to_)
    {
        next_layers_fmaps = this->network_layers[i]->get_feature_maps();
        next_layers_fmapcount = this->network_layers[i]->get_mapcount();
        next_layers_inputs = this->network_layers[i]->gets_input_from();
        /*if(next_layers_fmapcount != 1)
        {
            cerr << "Currently the fully connected layer can be followed only by fully connected layers!\n";
            throw exception();
        }*/
        for(j = 0; j < next_layers_inputs.size(); j++)
        {
            if(this->my_index == next_layers_inputs[j]) break;
        }

        get_fcc_delta(next_layers_fmaps[j][0].weights[0][0], deltas[i][0][0], this->output_derivative[threadindex][0][0], this->layers_delta[threadindex][0][0]);
    }
    nabla[0][0].biases[0][0] = this->layers_delta[threadindex][0][0];
    //nabla[0][0].weights[0][0] = this->layers_delta[threadindex][0][0] * input[0][0].transpose();
    l = 0;
    for(int k : this->gets_input_from_)
    {
        ///TODO nabla has to be planned
        nabla[l++][0].weights[0][0] = this->layers_delta[threadindex][0][0].multiply_with_transpose(this->network_layers[k]->get_output(threadindex)[0][0]);
    }
    deltas[this->my_index] = this->layers_delta[threadindex];
    return NULL;
}

inline int FullyConnected::get_threadcount()
{
    return this->threadcount;
}

void FullyConnected::set_threadcount(int threadcnt, vector<Matrix***> inputs_)
{
    this->destroy_dinamic_data();
    this->threadcount = threadcnt;
    this->build_dinamic_data();
    this->set_layers_inputs(inputs_);
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
    return this->inputs.size();
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
        cout << "WARNING: DROPOUT IS INACTIVATED!\n";
        probability = 0.0;
    }
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
        this->backup_activation_input = this->activation_input;
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
        this->activation_input = this->backup_activation_input;
        this->output_derivative = this->backup_output_derivative;
        this->output_error = this->backpup_output_error;
        this->output_error_helper = this->backup_output_error_helper;
        this->layers_delta = this->backup_layers_delta;
        this->outputlen = this->backup_outputlen;
    }
}

void FullyConnected::store(std::ofstream &params)
{
    for(int i = 0; i < this->inputs.size(); i++)
        this->fmap[i]->store(params);
}

void FullyConnected::load(std::ifstream &params)
{
    for(int i = 0; i < this->inputs.size(); i++)
        this->fmap[i]->load(params);
}

void FullyConnected::create_connections(vector<int> input_from, vector<int> output_to)
{
    ///TODO some error checking
    this->gets_input_from_ = input_from;
    this->sends_output_to_ = output_to;
}

const vector<int>& FullyConnected::gets_input_from() const
{
    return this->gets_input_from_;
}

const vector<int>& FullyConnected::sends_output_to() const
{
    return this->sends_output_to_;
}

void FullyConnected::set_graph_information(Layer **network_layers, int my_index)
{
    this->my_index = my_index;
    this->network_layers = network_layers;
}

void get_fcc_delta(Matrix &nextLW, Matrix &delta, Matrix &output_derivative, Matrix &ret)
{
    if(nextLW.row != delta.row)
    {
        throw std::invalid_argument("In the matrix multiplication the colums of lvalue must equal with the rows of rvalue!\n");
    }
    else
    {
        //Matrix ret(nextLW.col, delta.col);
        double c = 0;
        for(int k = 0; k < ret.row; k++)
        {
            for(int l = 0; l < delta.col; l++)
            {
                for(int i = 0; i < delta.row; i++)
                {
                    c += nextLW.data[i][k] * delta.data[i][l];
                    //c += data[k][i] * 1;
                }
                ret.data[k][l] += c * output_derivative.data[k][l];
                c = 0;
            }
        }
        //return ret;
    }
}

void weighted_output(Matrix &w, Matrix &input, Matrix &b, Matrix &mtx)
{
    if(w.col != input.row)
    {
        throw std::invalid_argument("In the matrix multiplication the colums of lvalue must equal with the rows of rvalue!\n");
    }
    else
    {
        //Matrix mtx(w.row, input.col);
        double c = 0;
        for(int k = 0; k < w.row; k++)
        {
            for(int l = 0; l < input.col; l++)
            {
                for(int i = 0; i < w.col; i++)
                {
                    c += w.data[k][i] * input.data[i][l];
                    //c += data[k][i] * 1;
                }
                mtx.data[k][l] += c+b.data[k][l];
                c = 0;
            }
        }
        //return mtx;
    }
}
