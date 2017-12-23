#include "layers.h"

LayerDescriptor::LayerDescriptor(int layer_type, int neuron_type, int neuron_count, int stride):
            layer_type(layer_type), neuron_count(neuron_count), neuron_type(neuron_type), stride(stride) {}
