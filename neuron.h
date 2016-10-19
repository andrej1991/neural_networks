#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#define SIGMOID 0
#define LINEAR 1

class Neuron{
    int neuron_type;
    inline double sigmoid(double *weights, double **inputs, double bias, int inputs_len);
    inline double sigmoid_derivate(double *weights, double **inputs, double bias, int inputs_len);
    inline double linear(double *weights, double **inputs, double bias, int inputs_len);
    inline double linear_derivate(double *weights, double **inputs, double bias, int inputs_len);
    public:
    Neuron(int neuron_type = SIGMOID);
    double neuron(double *weights, double **inputs, double bias, int inputs_len);
    double neuron_derivate(double *weights, double **inputs, double bias, int inputs_len);
    void test();
};


#endif // NEURON_H_INCLUDED
