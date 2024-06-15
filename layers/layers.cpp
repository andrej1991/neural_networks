#include "layers.h"

Layer::~Layer(){}


void Layer::set_threadcount(int threadcount){throw runtime_error("Unimplemented function: Layer::set_threadcount(int threadcount)\n");}

inline int Layer::get_threadcount(){throw runtime_error("Unimplemented function: Layer::get_threadcount()\n");}

void Layer::create_connections(vector<int> input_from, vector<int> output_to){throw runtime_error("Unimplemented function: Layer::input_from\n");}

const vector<int>& Layer::gets_input_from() const
    {throw runtime_error("Unimplemented function: Layer::gets_input_from\n");}

const vector<int>& Layer::sends_output_to() const
    {throw runtime_error("Unimplemented function: Layer::sends_output_to\n");}

void Layer::set_layers_inputs(vector<Matrix***> inputs_){throw runtime_error("Unimplemented function: Layer::set_layers_inputs\n");}

int Layer::get_vertical_stride(){throw runtime_error("Unimplemented function: Layer::get_vertical_stride\n");}

int Layer::get_horizontal_stride(){throw runtime_error("Unimplemented function: Layer::get_horizontal_stride\n");}
