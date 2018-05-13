#include <CL/cl.h>
#include <iostream>
#include "matrix.h"
#include "../opencl_setup.h"

using namespace std;

cl_program MatrixOperations::multiply_program;
cl_program MatrixOperations::matrice_add_program;
cl_program MatrixOperations::scalar_add_program;
cl_program MatrixOperations::hadamart_program;
cl_program MatrixOperations::transpose_program;
cl_program MatrixOperations::convolution_program;
cl_program MatrixOperations::fullconv_program;
cl_program MatrixOperations::sameconv_program;
cl_program MatrixOperations::multiply_with_transpose_program;
int MatrixOperations::instance_count=0;
int MatrixData::instancecount=0;


MatrixData::MatrixData(int r,int c) : data(NULL), cl_mem_inuse(false), is_cl_memcontent_valid(false)
{
    if(r >= 0)
        this->row = r;
    else
        this->row = 1;
    if(c >= 0)
        this->col = c;
    else
        this->col = 1;
    try
    {
        this->data = new float [r*c];
        for(int i = 0; i < r*c; i++)
            this->data[i] = 0;

    }
    catch(bad_alloc& ba)
    {
        std::cerr << "MatrixData::constructor: bad_alloc caught: " << ba.what() << std::endl;
        throw;
    }
    //this->cl_mem_inuse = false;
    //this->is_cl_memcontent_valid = false;
}

inline void MatrixData::destruct()
{
    if(this->data != NULL)
    {
        delete[] this->data;
        this->data = NULL;
    }
}

inline void MatrixData::equality(const MatrixData &mtx)
{
    this->row = mtx.row;
    this->col = mtx.col;
    try
    {
        data = new float[(this->row) * (this->col)];

    }
    catch(bad_alloc& ba)
    {
        cerr << "MatrixData::copyconstructor: bad_alloc caught: " << ba.what() << endl;
        throw;
    }
    for(int i = 0; i < (this->row) * (this->col); i++)
    {
        this->data[i] = mtx.data[i];
    }
}

MatrixData::~MatrixData()
{
    this->destruct();
    if(this->cl_mem_inuse)
    {
        clReleaseMemObject(this->cl_mem_obj);
        this->cl_mem_inuse = false;
    }
}

MatrixData::MatrixData(const MatrixData& mtx)
{
    this->equality(mtx);
    this->cl_mem_inuse = false;
    this->is_cl_memcontent_valid = false;

}
MatrixData& MatrixData::operator= (const MatrixData& mtx)
{
    if(this==&mtx) return *this;
    this->destruct();
    this->equality(mtx);
    return *this;

}

float* MatrixData::operator[](int r)
{
    return this->data + (this->col * r);
}

const float* MatrixData::operator[](int r) const
{
    return this->data + (this->col * r);
}

void MatrixData::copy_to_opencl_buffer(cl_context *context, cl_command_queue* comm_queue)
{
    cl_int errorcode;
    //cout << MatrixData::instancecount<<endl;
    //MatrixData::instancecount++;
    if((this->cl_mem_inuse) && (comm_queue!=NULL))
    {
        clEnqueueWriteBuffer(*comm_queue, this->cl_mem_obj, CL_TRUE, 0, row*col*sizeof(float), (void*)this->data, 0, NULL, NULL);
        return;
    }
    else
    {
        if(this->cl_mem_inuse)
        {
            clReleaseMemObject(this->cl_mem_obj);
        }
        this->cl_mem_obj = clCreateBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->row * this->col * sizeof(float), this->data, &errorcode);
        if(errorcode != CL_SUCCESS)
        {
            cerr << "unable to create OpenCL buffer\n";
            throw exception();
        }
        this->cl_mem_inuse = true;
        ///TODO increment the reference count of context
        context_of_cl_mem_obj = *context;
    }
}

int MatrixData::get_row()
{
    return this->row;
}

int MatrixData::get_col()
{
    return this->col;
}

MatrixOperations::MatrixOperations(cl_context *context, cl_device_id *deviceIds)
{
    if(MatrixOperations::instance_count == 0)
        this->load_matrice_operations_programs(context, deviceIds);
    instance_count++;
    cl_int errorcode;
    this->command_queue = clCreateCommandQueue(*context, deviceIds[0], 0, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL command queue\n";
        throw exception();
    }
    this->matrice_add_kernel = clCreateKernel(MatrixOperations::matrice_add_program, "add", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::matrice_add_program kernel\n";
        throw exception();
    }
    this->matrice_substract_kernel = clCreateKernel(MatrixOperations::matrice_add_program, "add", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::matrice_substract_program kernel\n";
        throw exception();
    }
    this->scalar_add_kernel = clCreateKernel(MatrixOperations::scalar_add_program, "scalar_matrice_add", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::scalar_add_program kernel\n";
        throw exception();
    }
    this->transpose_kernel = clCreateKernel(MatrixOperations::transpose_program, "transpose", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::transpose_program kernel\n";
        throw exception();
    }
    this->multiply_kernel = clCreateKernel(MatrixOperations::multiply_program, "multiply", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::multiply_program kernel\n";
        throw exception();
    }
    this->hadamart_kernel = clCreateKernel(MatrixOperations::hadamart_program, "hadamart_product", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::hadamart_program kernel\n";
        throw exception();
    }
    this->convolution_kernel = clCreateKernel(MatrixOperations::convolution_program, "convolution", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::convolution_program kernel\n";
        throw exception();
    }
    this->fullconv_kernel = clCreateKernel(MatrixOperations::fullconv_program, "fullconv", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::fullconv_program kernel\n";
        throw exception();
    }
    this->sameconv_kernel = clCreateKernel(MatrixOperations::sameconv_program, "sameconv", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::sameconv_program kernel\n";
        throw exception();
    }
    this->multiply_with_transpose_kernel = clCreateKernel(MatrixOperations::multiply_with_transpose_program, "multiply_with_transpose", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::multiply_with_transpose_program kernel\n";
        throw exception();
    }
    this->transpose_and_multiply_kernel = clCreateKernel(MatrixOperations::multiply_with_transpose_program, "multiply_with_transpose", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::transpose_and_multiply_program kernel\n";
        throw exception();
    }
    this->zero_kernel= clCreateKernel(MatrixOperations::multiply_with_transpose_program, "zero", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL MatrixOperations::zero kernel\n";
        throw exception();
    }
}

MatrixOperations::~MatrixOperations()
{
    clReleaseKernel(this->matrice_add_kernel);
    clReleaseKernel(this->scalar_add_kernel);
    clReleaseKernel(this->transpose_kernel);
    clReleaseKernel(this->multiply_kernel);
    clReleaseKernel(this->hadamart_kernel);
    clReleaseKernel(this->convolution_kernel);
    clReleaseKernel(this->fullconv_kernel);
    clReleaseKernel(this->sameconv_kernel);
    clReleaseKernel(this->multiply_with_transpose_kernel);
    clReleaseKernel(this->zero_kernel);
    clFlush(this->command_queue);
    clFinish(this->command_queue);
    clReleaseCommandQueue(this->command_queue);
    MatrixOperations::instance_count--;
    if(MatrixOperations::instance_count == 0)
    {
        clReleaseProgram(MatrixOperations::multiply_program);
        clReleaseProgram(MatrixOperations::matrice_add_program);
        clReleaseProgram(MatrixOperations::scalar_add_program);
        clReleaseProgram(MatrixOperations::hadamart_program);
        clReleaseProgram(MatrixOperations::transpose_program);
        clReleaseProgram(MatrixOperations::convolution_program);
        clReleaseProgram(MatrixOperations::fullconv_program);
        clReleaseProgram(MatrixOperations::sameconv_program);
        clReleaseProgram(MatrixOperations::multiply_with_transpose_program);
    }
}

void MatrixOperations::add_matrices(MatrixData &a, MatrixData &b, MatrixData &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.row) || (a.col != b.col) || (a.row != c.row) || (a.col != c.col))
    {
        cerr << "The addition is aborted because the matrices are not in the same size.\n";
        throw exception();
    }
    cl_int errorcode;
    size_t global_item_size = a.row*a.col;
    size_t local_item_size = a.row;
    errorcode = clSetKernelArg(this->matrice_add_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->matrice_add_kernel, 1, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clSetKernelArg(this->matrice_add_kernel, 2, sizeof(cl_mem), (void *)&(c.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->matrice_add_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);
}

void MatrixOperations::substract_matrices(MatrixData &a, MatrixData &b, MatrixData &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.row) || (a.col != b.col) || (a.row != c.row) || (a.col != c.col))
    {
        cerr << "The addition is aborted because the matrices are not in the same size.\n";
        throw exception();
    }
    cl_int errorcode;
    size_t global_item_size = a.row*a.col;
    size_t local_item_size = a.row;
    errorcode = clSetKernelArg(this->matrice_substract_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->matrice_substract_kernel, 1, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clSetKernelArg(this->matrice_substract_kernel, 2, sizeof(cl_mem), (void *)&(c.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->matrice_substract_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);
}

void MatrixOperations::scalar_add(MatrixData &a, float b, MatrixData &c, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != c.row) || (a.col != c.col))
    {
        cerr << "The addition is aborted because the matrices are not in the same size.\n";
        throw exception();
    }
    cl_int errorcode;
    size_t global_item_size = a.row*a.col;
    size_t local_item_size = a.col;
    errorcode = clSetKernelArg(this->scalar_add_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->scalar_add_kernel, 1, sizeof(float), (void *)&b);
    errorcode = clSetKernelArg(this->scalar_add_kernel, 2, sizeof(cl_mem), (void *)&(c.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->scalar_add_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);
}

void MatrixOperations::transpose(MatrixData &a, MatrixData &b, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.col) || (a.col != b.row))
    {
        cerr << "The \"transpose\" operation is aborted because the matrices are not in the required size.\n";
        throw exception();
    }
    cl_int errorcode;
    size_t global_item_size = a.row*a.col;
    size_t local_item_size = a.col;
    errorcode = clSetKernelArg(this->transpose_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->transpose_kernel, 1, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->transpose_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);
}

void MatrixOperations::zero(MatrixData &a, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    cl_int errorcode;
    size_t global_item_size = a.row*a.col;
    size_t local_item_size = a.row;
    errorcode = clSetKernelArg(this->zero_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->zero_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);
}

void MatrixOperations::multiply(MatrixData &a, MatrixData &b, MatrixData &c, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != c.row) || (a.col != b.row) || (b.col != c.col))
    {
        cerr << "The multiplication is aborted because the matrices are not in the required size.\n";
        throw exception();
    }
    cl_int errorcode;
    errorcode = clSetKernelArg(this->multiply_kernel, 0, sizeof(int), (void*)&a.row);
    errorcode = clSetKernelArg(this->multiply_kernel, 1, sizeof(int), (void*)&b.col);
    errorcode = clSetKernelArg(this->multiply_kernel, 2, sizeof(int), (void*)&b.row);
    errorcode = clSetKernelArg(this->multiply_kernel, 3, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->multiply_kernel, 4, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clSetKernelArg(this->multiply_kernel, 5, sizeof(cl_mem), (void *)&(c.cl_mem_obj));

    const size_t local[2] = { 2,4 };///TODO handle the local size
    const size_t global[2] = { (size_t)c.row, (size_t)c.col };
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->multiply_kernel, 2, NULL, global, local, num_events, wait_for_events, generated_event);
}

void MatrixOperations::multiply_with_transpose(MatrixData &a, MatrixData &b, MatrixData &c, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != c.row) || (a.col != b.col) || (b.row != c.col))
    {
        cerr << "The multiplication is aborted because the matrices are not in the required size.\n";
        throw exception();
    }
    cl_int errorcode;
    errorcode = clSetKernelArg(this->multiply_with_transpose_kernel, 0, sizeof(int), (void*)&a.row);
    errorcode = clSetKernelArg(this->multiply_with_transpose_kernel, 1, sizeof(int), (void*)&b.col);
    errorcode = clSetKernelArg(this->multiply_with_transpose_kernel, 2, sizeof(int), (void*)&b.row);
    errorcode = clSetKernelArg(this->multiply_with_transpose_kernel, 3, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->multiply_with_transpose_kernel, 4, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clSetKernelArg(this->multiply_with_transpose_kernel, 5, sizeof(cl_mem), (void *)&(c.cl_mem_obj));

    const size_t local[2] = { 2,4 };///TODO handle the local size
    const size_t global[2] = { (size_t)c.row, (size_t)c.col };
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->multiply_with_transpose_kernel, 2, NULL, global, local, num_events, wait_for_events, generated_event);
}

void MatrixOperations::transpose_and_multiply(MatrixData &a, MatrixData &b, MatrixData &c, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.col != c.row) || (a.row != b.row) || (b.col != c.col))
    {
        cerr << "The multiplication is aborted because the matrices are not in the required size.\n";
        throw exception();
    }
    cl_int errorcode;
    errorcode = clSetKernelArg(this->transpose_and_multiply_kernel, 0, sizeof(int), (void*)&a.row);
    errorcode = clSetKernelArg(this->transpose_and_multiply_kernel, 1, sizeof(int), (void*)&b.col);
    errorcode = clSetKernelArg(this->transpose_and_multiply_kernel, 2, sizeof(int), (void*)&b.row);
    errorcode = clSetKernelArg(this->transpose_and_multiply_kernel, 3, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->transpose_and_multiply_kernel, 4, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clSetKernelArg(this->transpose_and_multiply_kernel, 5, sizeof(cl_mem), (void *)&(c.cl_mem_obj));

    const size_t local[2] = { 2,4 };///TODO handle the local size
    const size_t global[2] = { (size_t)c.row, (size_t)c.col };
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->transpose_and_multiply_kernel, 2, NULL, global, local, num_events, wait_for_events, generated_event);
}

void MatrixOperations::hadamart(MatrixData &a, MatrixData &b, MatrixData &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.row) || (a.col != b.col) || (a.row != c.row) || (a.col != c.col))
    {
        cerr << "The calculation of the hadamart product is aborted because the matrices are not in the same size.\n";
        throw exception();
    }
    cl_int errorcode;
    size_t global_item_size = a.row*a.col;
    size_t local_item_size = a.row;
    errorcode = clSetKernelArg(this->hadamart_kernel, 0, sizeof(cl_mem), (void *)&(a.cl_mem_obj));
    errorcode = clSetKernelArg(this->hadamart_kernel, 1, sizeof(cl_mem), (void *)&(b.cl_mem_obj));
    errorcode = clSetKernelArg(this->hadamart_kernel, 2, sizeof(cl_mem), (void *)&(c.cl_mem_obj));
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->hadamart_kernel, 1, NULL, &global_item_size, &local_item_size, num_events, wait_for_events, generated_event);
}

void MatrixOperations::convolution(MatrixData &input, MatrixData &kernel, MatrixData &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    ///TODO some error checking
    cl_int errorcode;
    errorcode = clSetKernelArg(this->convolution_kernel, 0, sizeof(int), (void*)&kernel.row);
    errorcode = clSetKernelArg(this->convolution_kernel, 1, sizeof(int), (void*)&kernel.col);
    errorcode = clSetKernelArg(this->convolution_kernel, 2, sizeof(int), (void*)&input.col);
    errorcode = clSetKernelArg(this->convolution_kernel, 3, sizeof(int), (void*)&output.col);
    errorcode = clSetKernelArg(this->convolution_kernel, 4, sizeof(cl_mem), (void *)&(input.cl_mem_obj));
    errorcode = clSetKernelArg(this->convolution_kernel, 5, sizeof(cl_mem), (void *)&(kernel.cl_mem_obj));
    errorcode = clSetKernelArg(this->convolution_kernel, 6, sizeof(cl_mem), (void *)&(output.cl_mem_obj));
    const size_t global[2] = { (size_t)output.row, (size_t)output.col };
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->convolution_kernel, 2, NULL, global, NULL, num_events, wait_for_events, generated_event);
}

void MatrixOperations::fullconv(MatrixData &input, MatrixData &kernel, MatrixData &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    ///TODO some error checking
    cl_int errorcode;
    errorcode = clSetKernelArg(this->fullconv_kernel, 0, sizeof(int), (void*)&kernel.row);
    errorcode = clSetKernelArg(this->fullconv_kernel, 1, sizeof(int), (void*)&kernel.col);
    errorcode = clSetKernelArg(this->fullconv_kernel, 2, sizeof(int), (void*)&input.col);
    errorcode = clSetKernelArg(this->fullconv_kernel, 3, sizeof(int), (void*)&input.row);
    errorcode = clSetKernelArg(this->fullconv_kernel, 4, sizeof(int), (void*)&output.col);
    errorcode = clSetKernelArg(this->fullconv_kernel, 5, sizeof(cl_mem), (void *)&(input.cl_mem_obj));
    errorcode = clSetKernelArg(this->fullconv_kernel, 6, sizeof(cl_mem), (void *)&(kernel.cl_mem_obj));
    errorcode = clSetKernelArg(this->fullconv_kernel, 7, sizeof(cl_mem), (void *)&(output.cl_mem_obj));
    const size_t global[2] = { (size_t)output.row, (size_t)output.col };
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->fullconv_kernel, 2, NULL, global, NULL, num_events, wait_for_events, generated_event);
}

void MatrixOperations::sameconv(MatrixData &input, MatrixData &kernel, MatrixData &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    ///TODO some error checking
    cl_int errorcode;
    errorcode = clSetKernelArg(this->sameconv_kernel, 0, sizeof(int), (void*)&kernel.row);
    errorcode = clSetKernelArg(this->sameconv_kernel, 1, sizeof(int), (void*)&kernel.col);
    errorcode = clSetKernelArg(this->sameconv_kernel, 2, sizeof(int), (void*)&output.col);
    errorcode = clSetKernelArg(this->sameconv_kernel, 3, sizeof(cl_mem), (void *)&(input.cl_mem_obj));
    errorcode = clSetKernelArg(this->sameconv_kernel, 4, sizeof(cl_mem), (void *)&(kernel.cl_mem_obj));
    errorcode = clSetKernelArg(this->sameconv_kernel, 5, sizeof(cl_mem), (void *)&(output.cl_mem_obj));
    const size_t global[2] = { (size_t)output.row, (size_t)output.col };
    errorcode = clEnqueueNDRangeKernel(this->command_queue, this->sameconv_kernel, 2, NULL, global, NULL, num_events, wait_for_events, generated_event);
}

void print_mtx_list(MatrixData **mtx, int list_len)
{
    for(int i = 0; i < list_len; i++)
    {
        cout << "[";
        for(int j = 0; j < mtx[i][0].row; j++)
        {
            cout << "[";
            for(int k = 0; k < mtx[i][0].col; k++)
            {
                cout << mtx[i][0][j][k] << "; ";
            }
            cout << "]\n";
        }
        cout << "]\n";
    }

}

void print_mtx(MatrixData &mtx)
{
    cout << "[";
    for(int j = 0; j < mtx.row; j++)
    {
        cout << "[";
        for(int k = 0; k < mtx.col; k++)
        {
            cout << mtx[j][k] << "; ";
        }
        cout << "]\n";
    }
    cout << "]\n";
}


void MatrixOperations::load_matrice_operations_programs(cl_context *context, cl_device_id *deviceIds)
{
    MatrixOperations::matrice_add_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::scalar_add_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::transpose_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::multiply_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::hadamart_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::convolution_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::fullconv_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::sameconv_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
    MatrixOperations::multiply_with_transpose_program = load_program("matrix/opencl_kernels.cl",context, deviceIds);
}
