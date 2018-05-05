#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <CL/cl.h>

class MatrixData{
    public:
    int row, col;
    bool cl_mem_inuse;
    float *data;
    cl_mem cl_mem_obj;
    cl_context context_of_cl_mem_obj;
    MatrixData(int row=1, int col=1);
    ~MatrixData();
    MatrixData(const MatrixData& mtx);
    MatrixData operator= (const MatrixData& mtx);
    inline void destruct();
    inline void equality(const MatrixData &mtx);
    float* operator[](int r);
    const float* operator[](int r) const;
    void copy_to_opencl_buffer(cl_context *context);
    int get_row();
    int get_col();
};


class MatrixOperations{
    static int instance_count;
    public:
    static cl_program multiply_program;
    static cl_program matrice_add_program;
    static cl_program scalar_add_program;
    static cl_program hadamart_program;
    static cl_program transpose_program;
    static cl_program convolution_program;
    static cl_program fullconv_program;
    static cl_program sameconv_program;
    static cl_program multiply_with_transpose_program;
    cl_kernel multiply_kernel;
    cl_kernel transpose_and_multiply_kernel;
    cl_kernel matrice_add_kernel;
    cl_kernel matrice_substract_kernel;
    cl_kernel scalar_add_kernel;
    cl_kernel hadamart_kernel;
    cl_kernel transpose_kernel;
    cl_kernel convolution_kernel;
    cl_kernel fullconv_kernel;
    cl_kernel sameconv_kernel;
    cl_kernel multiply_with_transpose_kernel;

    cl_command_queue command_queue;

    MatrixOperations(cl_context *context, cl_device_id *deviceIds);
    ~MatrixOperations();
    void load_matrice_operations_programs(cl_context *context, cl_device_id *deviceIds);
    void add_matrices(MatrixData &a, MatrixData &b, MatrixData &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void substract_matrices(MatrixData &a, MatrixData &b, MatrixData &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void multiply(MatrixData &a, MatrixData &b, MatrixData &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void transpose_and_multiply(MatrixData &a, MatrixData &b, MatrixData &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void multiply_with_transpose(MatrixData &a, MatrixData &b, MatrixData &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void scalar_add(MatrixData &a, float b, MatrixData &c, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void transpose(MatrixData &a, MatrixData &b, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void hadamart(MatrixData &a, MatrixData &b, MatrixData &c,int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void convolution(MatrixData &input, MatrixData &kernel, MatrixData &output, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void fullconv(MatrixData &input, MatrixData &kernel, MatrixData &output, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void sameconv(MatrixData &input, MatrixData &kernel, MatrixData &output, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
    void sigmoid(MatrixData &input, MatrixData &output, int num_events=0, cl_event *wait_for_events=NULL, cl_event *generated_event=NULL);
};

void print_mtx_list(MatrixData **mtx, int list_len);
void print_mtx(MatrixData &mtx);

#endif // MATRIX_H_INCLUDED
