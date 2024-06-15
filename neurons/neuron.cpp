#include "neuron.h"
#include <math.h>
#include <iostream>

#define SIG(param)   (1 / (1 + exp(-1 * param)))

Neuron::Neuron(int neuron_type) : neuron_type(neuron_type){}

inline void Neuron::sigmoid(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            outputs.data[i][j] = SIG(inputs.data[i][j]);
        }
    }
}

inline void Neuron::sigmoid_derivative(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    double s;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            s = SIG(inputs.data[i][j]);
            outputs.data[i][j] = s * (1 - s);
        }
    }
}

inline void Neuron::tan_hip(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            outputs.data[i][j] = tanh(inputs.data[i][j]);
        }
    }
}

inline void Neuron::tan_hip_derivative(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    double s;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            s = tanh(inputs.data[i][j]);
            outputs.data[i][j] = 1 - s * s;
        }
    }
}

inline void Neuron::relu(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = inputs.data[i][j];
                    else
                        outputs.data[i][j] = 0;
                }
        }
}
inline void Neuron::relu_derivative(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = 1;
                    else
                        outputs.data[i][j] = 0;
                }
        }
}

inline void Neuron::leaky_relu(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = inputs.data[i][j];
                    else
                        outputs.data[i][j] = 0.001*inputs.data[i][j];
                }
        }
}
inline void Neuron::leaky_relu_derivative(Matrix &inputs, Matrix &outputs){
    int row = inputs.get_row();
    int col = inputs.get_col();
    for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                    if(inputs.data[i][j] > 0)
                        outputs.data[i][j] = 1;
                    else
                        outputs.data[i][j] = 0.001;
                }
        }
}

void Neuron::neuron(Matrix &inputs, Matrix &outputs){
    switch(this->neuron_type){
    case SIGMOID:
        this->sigmoid(inputs, outputs);
        return;
    case RELU:
        this->relu(inputs, outputs);
        return;
    case LEAKY_RELU:
        this->leaky_relu(inputs, outputs);
        return;
    case TANH:
        this->tan_hip(inputs, outputs);
        return;
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}

void Neuron::neuron_derivative(Matrix &inputs, Matrix &outputs){
    switch(this->neuron_type){
    case SIGMOID:
        this->sigmoid_derivative(inputs, outputs);
        return;
    case RELU:
        this->relu_derivative(inputs, outputs);
        return;
    case LEAKY_RELU:
        this->leaky_relu_derivative(inputs, outputs);
        return;
    case TANH:
        this->tan_hip_derivative(inputs, outputs);
        return;
    default:
        std::cerr << "Unknown neuron type;";
        throw std::exception();
    }
}



