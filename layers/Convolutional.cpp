#include "layers.h"

Convolutional::Convolutional(int input_row, int input_col, int input_channel_count, int kern_row, int kern_col, int map_count, int neuron_type, int next_layers_type, Padding &p, int stride):
                    input_row(input_row), input_col(input_col), kernel_row(kern_row), kernel_col(kern_col), map_count(map_count), stride(stride), next_layers_type(next_layers_type),
                    pad(p.left_padding, p.top_padding, p.right_padding, p.bottom_padding), neuron(neuron_type), neuron_type(neuron_type), tpool(map_count)
{
    if(stride != 1)
        {
            std::cerr << "counting with stride different than 1 is not implemented yet!";
            throw exception();
        }
    this->output_row = input_row - kern_row + 1;
    this->output_col = input_col - kern_col + 1;
    this->layer_type = CONVOLUTIONAL;
    this->fmap = new Feature_map* [map_count];
    this->outputs = new Matrice* [map_count];
    this->output_derivative = new Matrice* [map_count];
    this->layers_delta = new Matrice* [map_count];
    this->layers_delta_helper = new Matrice* [map_count];
    this->output_job = new JobInterface* [map_count];
    this->output_derivative_job = new JobInterface* [map_count];
    this->nabla_calculator_job = new JobInterface* [map_count];
    this->delta_helper_conv_job = new JobInterface* [map_count];
    this->delta_helper_nonconv_job = new JobInterface* [map_count];
    this->flattened_output = new Matrice* [1];
    this->flattened_output[0] = new Matrice(this->map_count * this->output_row * this->output_col, 1);
    for(int i = 0; i < map_count; i++)
    {
        fmap[i] = new Feature_map(this->kernel_row, this->kernel_col, input_channel_count);
        this->outputs[i] = new Matrice(this->output_row, this->output_col);
        this->output_derivative[i] = new Matrice(this->output_row, this->output_col);
        this->layers_delta[i] = new Matrice(this->output_row, this->output_col);
        this->layers_delta_helper[i] = new Matrice(this->output_row, this->output_col);
        this->output_job[i] = new GetOutputJob(i, this);
        this->output_derivative_job[i] = new GetOutputDerivativeJob(i, this);
        this->nabla_calculator_job[i] = new CalculatingNabla(i, this);
        this->delta_helper_conv_job[i] = new CalculatingDeltaHelperConv(i, this);
        this->delta_helper_nonconv_job[i] = new CalculatingDeltaHelperNonConv(i, this);
    }
}

Convolutional::~Convolutional()
{
    delete flattened_output[0];
    delete[] flattened_output;
    for(int i = 0; i < this->map_count; i++)
    {
        delete fmap[i];
        delete outputs[i];
        delete output_derivative[i];
        delete layers_delta[i];
        delete layers_delta_helper[i];
    }
    delete[] fmap;
    delete[] outputs;
    delete[] output_derivative;
    delete[] layers_delta;
    delete[] layers_delta_helper;
}

void Convolutional::get_2D_weights(int neuron_id, int fmap_id, Matrice &kernel, Feature_map **next_layers_fmap)
{
    int kernelsize = kernel.get_row() * kernel.get_col();
    int starting_pos = kernelsize * fmap_id;
    int endpos = starting_pos + kernelsize;
    int index = starting_pos;
    for(int col = 0; col < kernel.get_col(); col++)
    {
        for(int row = 0; row < kernel.get_row(); row++)
        {
            kernel.data[row][col] = next_layers_fmap[0]->weights[0]->data[neuron_id][index];
            index++;
        }
    }
}

inline void calculate_delta_helper(Matrice *padded_delta, Matrice *delta_helper, Matrice &kernel, Matrice &helper)
{
    convolution(padded_delta[0],kernel, helper);
    delta_helper[0] += helper;
}

inline void delete_padded_delta(Matrice **padded_delta, int limit)
{
    for(int i = 0; i < limit; i++)
    {
        delete padded_delta[i];
    }
    delete[] padded_delta;
}

inline Matrice** Convolutional::backpropagate(Matrice **input, Feature_map** next_layers_fmaps, Feature_map** nabla, Matrice **delta, int next_layers_fmapcount)
{
    this->derivate_layers_output(input);
    Matrice **padded_delta;
    Matrice helper(this->output_row, this->output_col);
    //JobInterface **nabla_calculator_job;
    //nabla_calculator_job = new JobInterface*[this->map_count];
    if(this->next_layers_type != CONVOLUTIONAL)
    {
        int next_layers_neuroncount = delta[0]->get_row();
        //padded_delta_job = new JobInterface*[next_layers_neuroncount];
        padded_delta = new Matrice* [next_layers_neuroncount];
        for(int i = 0; i < next_layers_neuroncount; i++)
        {
            padded_delta[i] = new Matrice;
            padded_delta[i][0].data[0][0] = delta[0][0].data[i][0];
            padded_delta[i][0] = padded_delta[i][0].zero_padd((this->output_row-1)/2,
                                                     (this->output_col-1)/2,
                                                     (this->output_row-1)/2,
                                                     (this->output_col-1)/2);
        }
        /*for(int i = 0; i < next_layers_neuroncount; i++)
        {
            padded_delta_job[i] = new GetPaddedDeltaNonConv(i, (this->output_row-1)/2,
                                                     (this->output_col-1)/2,
                                                     (this->output_row-1)/2,
                                                     (this->output_col-1)/2,  this, padded_delta, delta);
            tpool.push(padded_delta_job[i]);
        }
        tpool.wait();*/
        for(int i = 0; i < this->map_count; i++)
        {
            delta_helper_nonconv_job[i]->set_params_for_deltahelper(next_layers_neuroncount, next_layers_fmaps, padded_delta);
            tpool.push(delta_helper_nonconv_job[i]);
        }
        tpool.wait();
        delete_padded_delta(padded_delta, next_layers_neuroncount);
    }
    else
    {
        padded_delta = new Matrice* [next_layers_fmapcount];
        for(int i = 0; i < next_layers_fmapcount; i++)
        {
            padded_delta[i] = new Matrice;
            padded_delta[i][0] = delta[i][0];
            padded_delta[i][0] = delta[i][0].zero_padd((next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_row()-1)/2,
                                                     (next_layers_fmaps[i]->weights[0]->get_col()-1)/2);
        }
        /*for(int i = 0; i < next_layers_fmapcount; i++)
        {
            padded_delta_job[i] = new GetPaddedDeltaConv(i, int((next_layers_fmaps[i]->weights[0]->get_row()-1)/2),
                                                     int((next_layers_fmaps[i]->weights[0]->get_col()-1)/2),
                                                     int((next_layers_fmaps[i]->weights[0]->get_row()-1)/2),
                                                     int((next_layers_fmaps[i]->weights[0]->get_col()-1)/2),  this, padded_delta, delta);
            tpool.push(padded_delta_job[i]);
        }
        tpool.wait();*/
        for(int i = 0; i < this->map_count; i++)
        {
            delta_helper_conv_job[i]->set_params_for_deltahelper(next_layers_fmapcount, next_layers_fmaps, padded_delta);
            tpool.push(delta_helper_conv_job[i]);
        }
        tpool.wait();
        delete_padded_delta(padded_delta, next_layers_fmapcount);
    }
    for(int i = 0; i < this->map_count; i++)
    {
        //nabla_calculator_job[i] = new CalculatingNabla(i, this, input, nabla);
        dynamic_cast<CalculatingNabla*>(nabla_calculator_job[i])->input = input;
        dynamic_cast<CalculatingNabla*>(nabla_calculator_job[i])->nabla = nabla;
        tpool.push(nabla_calculator_job[i]);
    }
    tpool.wait();
    /*for(int i = 0; i < this->map_count; i++)
    {
        delete nabla_calculator_job[i];
        delete delta_helper_conv_job[i];
        //delete padded_delta_job[i];

    }
    delete[] nabla_calculator_job;
    delete[] delta_helper_conv_job;
    //delete[] padded_delta_job;*/
    return this->layers_delta;
}

void Convolutional::update_weights_and_biasses(double learning_rate, double regularization_rate, Layers_features *layer)
{
    for(int i = 0; i < this->map_count; i++)
    {
        for(int j = 0; j < this->fmap[i]->get_mapdepth(); j++)
        {
            for(int row = 0; row < this->kernel_row; row++)
            {
                for(int col = 0; col < this->kernel_col; col++)
                {
                    this->fmap[i]->weights[j]->data[row][col] =
                                    regularization_rate * this->fmap[i]->weights[j]->data[row][col] -
                                    learning_rate * layer->fmap[i]->weights[j]->data[row][col];
                }
            }
        }
    }
}

inline void Convolutional::fulldepth_conv(Matrice &helper, Matrice &convolved, Matrice **input, int map_index)
{
    for(int channel_index = 0; channel_index < this->fmap[map_index]->get_mapdepth(); channel_index++)
    {
        convolution(input[channel_index][0], this->fmap[map_index]->weights[channel_index][0], convolved, this->stride);
        helper += convolved;
    }
    helper+=this->fmap[map_index]->biases[0][0].data[0][0];
}

inline void Convolutional::layers_output(Matrice **input)
{
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        //output_job[map_index]->set_input(input);
        dynamic_cast<GetOutputJob*>(output_job[map_index])->input = input;
        tpool.push(output_job[map_index]);
    }
    tpool.wait();
}

inline Matrice** Convolutional::get_output_error(Matrice **input, Matrice &required_output, int costfunction_type)
{
    cerr << "currently the convolutional neural network needs to have atleest one fully connected layer at the output";
    throw exception();
}

inline Matrice** Convolutional::derivate_layers_output(Matrice **input)
{
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        //output_derivative_job[map_index]->set_input(input);
        dynamic_cast<GetOutputDerivativeJob*>(output_derivative_job[map_index])->input = input;
        tpool.push(output_derivative_job[map_index]);
    }
    tpool.wait();
    return output_derivative;
}

void Convolutional::flatten()
{
    ///TODO rewrite this if the feature maps can have different kernel size;
    int i = 0;
    for(int map_index = 0; map_index < this->map_count; map_index++)
    {
        for(int col = 0; col < this->output_col; col++)
        {
            for(int row = 0; row < this->output_row; row++)
            {
                this->flattened_output[0]->data[i][0] = this->outputs[map_index]->data[row][col];
                i++;
            }
        }
    }
}


inline void Convolutional::remove_some_neurons(Matrice ***w_bckup, Matrice ***b_bckup, int **layers_bckup, int ***indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

inline void Convolutional::add_back_removed_neurons(Matrice **w_bckup, Matrice **b_bckup, int *layers_bckup, int **indexes)
{
    ;///this function doesn't have meaning in convolutional layer; it's only here for interface compatibility
}

void Convolutional::set_input(Matrice **input)
{
    cerr << "This function can be called only for the InputLayer!\n";
    throw exception();
}

inline Matrice** Convolutional::get_output()
{
    //print_mtx_list(outputs, this->map_count);
    //cout << "---------------------------" << endl;
    if(next_layers_type == FULLY_CONNECTED)
    {
        this->flatten();
        return this->flattened_output;
    }
    else
        return this->outputs;
}

inline Feature_map** Convolutional::get_feature_maps()
{
    return this->fmap;
}

inline short Convolutional::get_layer_type()
{
    return CONVOLUTIONAL;
}

inline int Convolutional::get_output_row()
{
    return this->output_row;
}

inline int Convolutional::get_output_len()
{
    return (this->output_row * this->output_col * this->map_count);
}

inline int Convolutional::get_output_col()
{
    return this->output_col;
}

void Convolutional::set_weights(Matrice *w)
{
    ;
}

void Convolutional::set_biases(Matrice *b)
{
    ;
}

int Convolutional::get_mapcount()
{
    return this->map_count;
}

int Convolutional::get_mapdepth()
{
    return this->fmap[0]->get_mapdepth();
}

int Convolutional::get_weights_row()
{
    return this->kernel_row;
}

int Convolutional::get_weights_col()
{
    return this->kernel_col;
}

void Convolutional::store(std::ofstream &params)
{
    for(int i = 0; i < this->map_count; i++)
    {
        this->fmap[i]->store(params);
    }
}
void Convolutional::load(std::ifstream &params)
{
    for(int i = 0; i < this->map_count; i++)
    {
        this->fmap[i]->load(params);
    }
}
