#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#define SIGMOID 0

class Neuron{
    public:
    double sigmoid(double *weights, double **inputs, double bias, int inputs_len);
    double sigmoid_derivate(double *weights, double **inputs, double bias, int inputs_len);
    void test();
};


#endif // NEURON_H_INCLUDED
