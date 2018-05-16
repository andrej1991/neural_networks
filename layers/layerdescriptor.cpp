#include "layers.h"
#include "../opencl_setup.h"

LayerDescriptor::LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int col, int mapcount, int stride):
            layer_type(layer_type), neuron_count(neuron_count), neuron_type(neuron_type), stride(stride),
            row(neuron_count), col(col), mapcount(mapcount) {}


Layers_features::Layers_features(int mapcount, int row, int col, int depth, int biascnt, OpenclSetup *env):
            fmap_count(mapcount)
{
    this->fmap = new Feature_map* [this->fmap_count];
    for(int i = 0; i < mapcount; i++)
        {
            this->fmap[i] = new Feature_map(row, col, depth, biascnt, env, false);
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

void Layers_features::operator+= (Layers_features &layer)
{
    for(int map_index = 0; map_index < this->fmap_count; map_index++)
    {
        int mapdepth = this->fmap[map_index][0].get_mapdepth();
        cl_event events[2*mapdepth];
        for(int i = 0; i < mapdepth; i++)
        {
            layer.fmap[map_index][0].mtxop[0].add_matrices(this->fmap[map_index][0].weights[i][0],
                                                           layer.fmap[map_index][0].weights[i][0],
                                                           this->fmap[map_index][0].weights[i][0],
                                                           //0, NULL, &events[2*i]);
                                                           0, NULL, NULL);
            layer.fmap[map_index][0].mtxop[0].add_matrices(this->fmap[map_index][0].biases[i][0],
                                                           layer.fmap[map_index][0].biases[i][0],
                                                           this->fmap[map_index][0].biases[i][0],
                                                           //0, NULL, &events[2*i+1]);
                                                           0, NULL, NULL);
        }
         //clWaitForEvents(2*mapdepth, events);
         clFinish(layer.fmap[map_index][0].mtxop[0].command_queue);
    }
}

int Layers_features::get_fmap_count()
{
    return this->fmap_count;
}
