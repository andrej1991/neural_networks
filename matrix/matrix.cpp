#include "matrix.h"
#include <CL/cl.h>

cl_program MatrixOperations::multiply_program;
cl_program MatrixOperations::matrice_add_program;
cl_program MatrixOperations::scalar_add_program;
cl_program MatrixOperations::hadamart_program;
cl_program MatrixOperations::transpose_program;
cl_program MatrixOperations::rot180_program;
cl_program MatrixOperations::zeropadd_program;
cl_program MatrixOperations::convolution_program;
cl_program MatrixOperations::fullconv_program;
cl_program MatrixOperations::sameconv_program;


MatrixData::MatrixData(int r,int c) : data(NULL)
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
        cerr << "MatrixData::constructor: bad_alloc caught: " << ba.what() << endl;
        throw;
    }
    this->cl_mem_inuse = false;
}

inline void MatrixData::destruct()
{
    if(this->data != NULL)
    {
        delete[] this->data;
        this->data = NULL;
    }
    if(this->cl_mem_inuse)
        clReleaseMemObject(this->cl_mem_obj);
        ///TODO release cl_context
}

inline void MatrixData::equality(const MatrixData &mtx)
{
    row = mtx.row;
    col = mtx.col;
    try
    {
        data = new float[this->row * this->col];

    }
    catch(bad_alloc& ba)
    {
        cerr << "MatrixData::copyconstructor: bad_alloc caught: " << ba.what() << endl;
        throw;
    }
    this->cl_mem_inuse = false;///mtx.cl_mem_inuse;
    if(this->cl_mem_inuse)
    {
        ///TODO hadle the memory copy from opencl device
        /*cl_uint errorcode;
        cl_command_queue command_queue = clCreateCommandQueue(mtx.context_of_cl_mem_obj, deviceIds[0], 0, &errorcode);
        if(errorcode != CL_SUCCESS)
        {
            cerr << "unable to create OpenCL command queue\n";
            throw exception();
        }
        errorcode = clEnqueueReadBuffer(command_queue, mtx.cl_mem_obj, CL_TRUE, 0, mtx.row * mtx.col * sizeof(float), this->data, 1, &event, NULL);
        this->cl_mem_obj = clCreateBuffer(mtx.context_of_cl_mem_obj, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->row * this->col * sizeof(float), this->data, &errorcode);
        this->context_of_cl_mem_obj = mtx.context_of_cl_mem_obj;
        clReleaseCommandQueue(command_queue);*/
    }
    else
    {
        for(int i = 0; i < row*col; i++)
        {
            this->data[i] = mtx.data[i];
        }
    }
    this->cl_mem_inuse = mtx.cl_mem_inuse;
}

MatrixData::~MatrixData()
{
    this->destruct();
}

MatrixData::MatrixData (const MatrixData& mtx)
{
    //this->destruct();
    this->equality(mtx);

}
MatrixData MatrixData::operator= (const MatrixData& mtx)
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

void MatrixData::create_opencl_buffer(cl_context *context)
{
    cl_int errorcode;
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
    clReleaseKernel(this->sigmoid_kernel);
    clReleaseProgram(this->matrice_add_program);
    clReleaseProgram(this->scalar_add_program);
    clReleaseProgram(this->transpose_program);
    clReleaseProgram(this->multiply_program);
    clReleaseProgram(this->hadamart_program);
    clReleaseProgram(this->convolution_program);
    clReleaseProgram(this->fullconv_program);
    clReleaseProgram(this->sameconv_program);
    clReleaseProgram(this->sigmoid_program);
    clFlush(this->command_queue);
    clFinish(this->command_queue);
    clReleaseCommandQueue(this->command_queue);
}

void MatrixOperations::add_matrices(matrice_data &a, matrice_data &b, matrice_data &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::scalar_add(matrice_data &a, float b, matrice_data &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::transpose(matrice_data &a, matrice_data &b, int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::multiply(matrice_data &a, matrice_data &b, matrice_data &c, int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::hadamart(matrice_data &a, matrice_data &b, matrice_data &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::convolution(matrice_data &input, matrice_data &kernel, matrice_data &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::fullconv(matrice_data &input, matrice_data &kernel, matrice_data &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
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

void MatrixOperations::sameconv(matrice_data &input, matrice_data &kernel, matrice_data &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
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
