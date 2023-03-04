#include "layers.h"

LayerDescriptor::LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int col, int mapcount, int vertical_stride, int horizontal_stride):
            layer_type(layer_type), neuron_count(neuron_count), neuron_type(neuron_type), vertical_stride(vertical_stride), horizontal_stride(horizontal_stride),
            row(neuron_count), col(col), mapcount(mapcount) {}


Layers_features::Layers_features(int mapcount, int row, int col, int depth, int biasrow, int biascol):
            fmap_count(mapcount), biasrow(biasrow), biascol(biascol)
{
    this->fmap = new Feature_map* [this->fmap_count];
    for(int i = 0; i < mapcount; i++)
    {
        this->fmap[i] = new Feature_map(row, col, depth, biasrow, biascol);
    }
}

Layers_features::Layers_features(const Layers_features &layer)
{
    int row, col, depth;
    row = layer.fmap[0]->weights[0][0].get_row();
    col = layer.fmap[0]->weights[0][0].get_col();
    depth = layer.fmap[0]->get_mapdepth();
    this->fmap = new Feature_map* [this->fmap_count];
    for(int i = 0; i < this->fmap_count; i++)
    {
        this->fmap[i] = new Feature_map(row, col, depth, layer.biasrow, layer.biascol);
    }
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            this->fmap[map_index]->weights[i][0] = layer.fmap[map_index]->weights[i][0];
        }
        this->fmap[map_index]->biases[0][0] = layer.fmap[map_index]->biases[0][0];
    }
}

Layers_features::~Layers_features()
{
    for(int i = 0; i < this->fmap_count; i++)
    {
        delete this->fmap[i];
    }
    delete[] this->fmap;
}

Layers_features & Layers_features::operator= (const Layers_features &layer)
{
    for(int i = 0; i < this->fmap_count; i++)
    {
        delete this->fmap[i];
    }
    delete[] this->fmap;
    int row, col, depth;
    row = layer.fmap[0]->weights[0][0].get_row();
    col = layer.fmap[0]->weights[0][0].get_col();
    depth = layer.fmap[0]->get_mapdepth();
    this->fmap = new Feature_map* [this->fmap_count];
    for(int i = 0; i < this->fmap_count; i++)
    {
        this->fmap[i] = new Feature_map(row, col, depth, layer.biasrow, layer.biascol);
    }

    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            this->fmap[map_index]->weights[i][0] = layer.fmap[map_index]->weights[i][0];
        }
        this->fmap[map_index]->biases[0][0] = layer.fmap[map_index]->biases[0][0];
    }
    return *this;
}

void Layers_features::operator+= (const Layers_features &layer)
{
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            this->fmap[map_index]->weights[i][0] += layer.fmap[map_index]->weights[i][0];
        }
        this->fmap[map_index]->biases[0][0] += layer.fmap[map_index]->biases[0][0];
    }
}

Layers_features Layers_features::operator+(const Layers_features &layer)
{
    Layers_features new_layer(this->fmap_count, this->fmap[0]->get_row(), this->fmap[0]->get_col(), this->fmap[0]->get_mapdepth(), this->biasrow, this->biascol);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            new_layer.fmap[map_index]->weights[i][0] = this->fmap[map_index]->weights[i][0] + layer.fmap[map_index]->weights[i][0];
        }
        new_layer.fmap[map_index]->biases[0][0] = this->fmap[map_index]->biases[0][0] + layer.fmap[map_index]->biases[0][0];
    }
    return new_layer;
}

Layers_features Layers_features::operator/(const Layers_features &layer)
{
    Layers_features new_layer(this->fmap_count, this->fmap[0]->get_row(), this->fmap[0]->get_col(), this->fmap[0]->get_mapdepth(), this->biasrow, this->biascol);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            new_layer.fmap[map_index]->weights[i][0] = this->fmap[map_index]->weights[i][0] / layer.fmap[map_index]->weights[i][0];
        }
        new_layer.fmap[map_index]->biases[0][0] = this->fmap[map_index]->biases[0][0] / layer.fmap[map_index]->biases[0][0];
    }
    return new_layer;
}

Layers_features Layers_features::operator*(double d)
{
    Layers_features new_layer(this->fmap_count, this->fmap[0]->get_row(), this->fmap[0]->get_col(), this->fmap[0]->get_mapdepth(), this->biasrow, this->biascol);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            new_layer.fmap[map_index]->weights[i][0] = this->fmap[map_index]->weights[i][0] * d;
        }
        new_layer.fmap[map_index]->biases[0][0] = this->fmap[map_index]->biases[0][0] * d;
    }
    return new_layer;
}

Layers_features Layers_features::operator+(double d)
{
    Layers_features new_layer(this->fmap_count, this->fmap[0]->get_row(), this->fmap[0]->get_col(), this->fmap[0]->get_mapdepth(), this->biasrow, this->biascol);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            new_layer.fmap[map_index]->weights[i][0] = this->fmap[map_index]->weights[i][0] + d;
        }
        new_layer.fmap[map_index]->biases[0][0] = this->fmap[map_index]->biases[0][0] + d;
    }
    return new_layer;
}

Layers_features Layers_features::sqroot()
{
    Layers_features new_layer(this->fmap_count, this->fmap[0]->get_row(), this->fmap[0]->get_col(), this->fmap[0]->get_mapdepth(), this->biasrow, this->biascol);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            new_layer.fmap[map_index]->weights[i][0] = this->fmap[map_index]->weights[i][0].sqroot();
        }
        new_layer.fmap[map_index]->biases[0][0] = this->fmap[map_index]->biases[0][0].sqroot();
    }
    return new_layer;
}

Layers_features Layers_features::square_element_by()
{
    Layers_features new_layer(this->fmap_count, this->fmap[0]->get_row(), this->fmap[0]->get_col(), this->fmap[0]->get_mapdepth(), this->biasrow, this->biascol);
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index]->get_mapdepth();
        for(int i = 0; i < mapdepth; i++)
        {
            new_layer.fmap[map_index]->weights[i][0] = this->fmap[map_index]->weights[i][0].square_element_by();
        }
        new_layer.fmap[map_index]->biases[0][0] = this->fmap[map_index]->biases[0][0].square_element_by();
    }
    return new_layer;
}

void Layers_features::zero()
{
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
        {
            int mapdepth = this->fmap[map_index]->get_mapdepth();
            this->fmap[map_index]->biases[0][0].zero();
            for(int i = 0; i < mapdepth; i++)
                {
                    this->fmap[map_index]->weights[i][0].zero();
                    //this->fmap[map_index]->biases[i][0].zero();
                }
        }
}

int Layers_features::get_fmap_count()
{
    return this->fmap_count;
}
