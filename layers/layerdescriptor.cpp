#include "layers.h"

LayerDescriptor::LayerDescriptor(int layer_type, int neuron_count, int neuron_type, int stride):
            layer_type(layer_type), neuron_count(neuron_count), neuron_type(neuron_type), stride(stride) {}

inline int LayerDescriptor::get_layer_type()
{
    return this->layer_type;
}

inline int LayerDescriptor::get_neuron_count()
{
    return this->neuron_count;
}

inline int LayerDescriptor::get_stride()
{
    return this->stride;
}

inline int LayerDescriptor::get_neuron_type()
{
    return this->neuron_type;
}
