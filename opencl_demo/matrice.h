#ifndef MATRICE_H_INCLUDED
#define MATRICE_H_INCLUDED

#include <CL/cl.h>

class matrice_data{
    public:
    int row, col;
    bool cl_mem_inuse;
    float *data;
    cl_mem cl_mem_obj;
    matrice_data(int row, int col);
    ~matrice_data();
    matrice_data(const matrice_data& mtx);
    matrice_data operator= (const matrice_data& mtx);
    inline void destruct();
    inline void equality(const matrice_data &mtx);
    float* operator[](int r);
    const float* operator[](int r) const;
    matrice_data operator* (const matrice_data& other);
    void create_opencl_buffer(cl_context context);
    //void convolution(matrice_data &input, matrice_data &kernel, matrice_data &output, int stride=1);
};


class matrice_operations{
    public:
    static cl_program multiply_program;
    static cl_program matrice_add_program;
    static cl_program scalar_add_program;
    static cl_program hadamart_program;
    static cl_program transpose_program;
    static cl_program rot180_program;
    static cl_program zeropadd_program;
    static cl_program convolution_program;
    cl_kernel multiply_kernel;
    cl_kernel matrice_add_kernel;
    cl_kernel scalar_add_kernel;
    cl_kernel hadamart_kernel;
    cl_kernel transpose_kernel;
    cl_kernel rot180_kernel;
    cl_kernel zeropadd_kernel;
    cl_kernel convolution_kernel;

    cl_command_queue command_queue;

    matrice_operations(cl_context *context, cl_device_id *deviceIds);
    ~matrice_operations();
    void add_matrices(matrice_data &a, matrice_data &b, matrice_data &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void multiply(matrice_data &a, matrice_data &b, matrice_data &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void scalar_add(matrice_data &a, float b, matrice_data &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void transpose(matrice_data &a, matrice_data &b, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void hadamart(matrice_data &a, matrice_data &b, matrice_data &c,int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void convolution(matrice_data &input, matrice_data &kernel, matrice_data &output, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
};


#endif // MATRICE_H_INCLUDED
