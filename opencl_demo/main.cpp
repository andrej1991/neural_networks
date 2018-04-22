#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "matrice.h"

#include <CL/cl.h>

using namespace std;


cl_program matrice_operations::multiply_program;
cl_program matrice_operations::matrice_add_program;
cl_program matrice_operations::scalar_add_program;
cl_program matrice_operations::hadamart_program;
cl_program matrice_operations::transpose_program;
cl_program matrice_operations::rot180_program;
cl_program matrice_operations::zeropadd_program;
cl_program matrice_operations::convolution_program;


matrice_data::matrice_data(int row, int col):row(row),col(col)
{
    this->data = new float[row*col];
    for(int i = 0; i < row*col; i++)
        this->data[i] = 0;
    this->cl_mem_inuse = false;
}
matrice_data::~matrice_data()
{
    delete this->data;
    //if(this->cl_mem_inuse)
    //    clReleaseMemObject(this->cl_mem_obj);
}
float* matrice_data::operator[](int r)
{
    return this->data + (this->col * r);
}
const float* matrice_data::operator[](int r) const
{
    return this->data + (this->col * r);
}
void matrice_data::create_opencl_buffer(cl_context context)
{
    cl_int errorcode;
    this->cl_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this->row * this->col * sizeof(float), this->data, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL buffer\n";
        throw exception();
    }
}


matrice_operations::matrice_operations(cl_context *context, cl_device_id *deviceIds)
{
    cl_int errorcode;
    this->command_queue = clCreateCommandQueue(*context, deviceIds[0], 0, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL command queue\n";
        throw exception();
    }
    this->matrice_add_kernel = clCreateKernel(matrice_operations::matrice_add_program, "add", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL matrice_operations::matrice_add_program kernel\n";
        throw exception();
    }
    this->scalar_add_kernel = clCreateKernel(matrice_operations::scalar_add_program, "scalar_matrice_add", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL matrice_operations::scalar_add_program kernel\n";
        throw exception();
    }
    this->transpose_kernel = clCreateKernel(matrice_operations::transpose_program, "transpose", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL matrice_operations::transpose_program kernel\n";
        throw exception();
    }
    this->multiply_kernel = clCreateKernel(matrice_operations::multiply_program, "multiply", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL matrice_operations::multiply_program kernel\n";
        throw exception();
    }
    this->hadamart_kernel = clCreateKernel(matrice_operations::hadamart_program, "hadamart_product", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL matrice_operations::hadamart_program kernel\n";
        throw exception();
    }
    this->convolution_kernel = clCreateKernel(matrice_operations::convolution_program, "convolution", &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL matrice_operations::convolution_program kernel\n";
        throw exception();
    }
}

matrice_operations::~matrice_operations()
{
    clReleaseKernel(this->matrice_add_kernel);
    clFlush(this->command_queue);
    clFinish(this->command_queue);
    clReleaseCommandQueue(this->command_queue);
}

void matrice_operations::add_matrices(matrice_data &a, matrice_data &b, matrice_data &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.row) && (a.col != b.col) && (a.row != c.row) && (a.col != c.col))
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

void matrice_operations::scalar_add(matrice_data &a, float b, matrice_data &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != c.row) && (a.col != c.col))
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

void matrice_operations::transpose(matrice_data &a, matrice_data &b, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.col) && (a.col != b.row))
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

void matrice_operations::multiply(matrice_data &a, matrice_data &b, matrice_data &c, int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != c.row) && (a.col != b.row) && (b.col != c.col))
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

void matrice_operations::hadamart(matrice_data &a, matrice_data &b, matrice_data &c,int num_events, cl_event *wait_for_events, cl_event *generated_event)
{
    if((a.row != b.row) && (a.col != b.col) && (a.row != c.row) && (a.col != c.col))
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

void matrice_operations::convolution(matrice_data &input, matrice_data &kernel, matrice_data &output, int num_events, cl_event *wait_for_events, cl_event *generated_event)
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


cl_program load_program(char* kernel_source, cl_context *context, cl_device_id *deviceIds)
{
    cl_int errorcode;
    ifstream srcFile (kernel_source, ifstream::in);
    string srcProg(istreambuf_iterator<char>(srcFile),(istreambuf_iterator<char>()));
    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    cl_program program = clCreateProgramWithSource(*context, 1, &src, &length, &errorcode);
    errorcode |= clBuildProgram(program, 1, deviceIds, NULL, NULL, NULL);
    if(errorcode != CL_SUCCESS)
        {
            cerr << "unable to create OpenCL program\n";
            throw exception();
        }
    return program;
}

void load_matrice_operations_programs(cl_context *context, cl_device_id *deviceIds)
{
    matrice_operations::matrice_add_program = load_program("matrice_add_kernel.cl",context, deviceIds);
    matrice_operations::scalar_add_program = load_program("scalar_matrice_add_kernel.cl",context, deviceIds);
    matrice_operations::transpose_program = load_program("transpose.cl",context, deviceIds);
    matrice_operations::multiply_program = load_program("multiply.cl",context, deviceIds);
    matrice_operations::hadamart_program = load_program("hadamart_product.cl",context, deviceIds);
    matrice_operations::convolution_program = load_program("convolution.cl",context, deviceIds);
}


void print_opencl_platform_info(cl_platform_id &platformid, cl_platform_info name)
{
    cl_int errnum;
    size_t s;
    errnum = clGetPlatformInfo(platformid, name, 0, NULL, &s);
    if(errnum != CL_SUCCESS)
    {
        cerr << "unable to get OpenCL platform info\n";
        throw exception();
    }
    char * info = (char *)alloca(sizeof(char) * s);
    errnum =clGetPlatformInfo(platformid, name, s, info, NULL);
    if(errnum != CL_SUCCESS)
    {
        cerr << "unable to get OpenCL platform info\n";
        throw exception();
    }
    cout << info << endl;
}

cl_device_id* get_opencl_devices(cl_device_type device_type, cl_platform_id &platform, cl_uint &numDevices)
{
    cl_int errNum;
    errNum = clGetDeviceIDs(platform, device_type, 0, NULL, &numDevices);
    if (numDevices < 1)
    {
        std::cout << "No device found for platform "<< platform << std::endl;
        exit(1);
    }
    cl_device_id *deviceIds = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(platform, device_type, numDevices, deviceIds, NULL);
    if(errNum != CL_SUCCESS)
    {
        cerr << "unable to find devices for the platform\n";
        throw exception();
    }
    return deviceIds;
}

cl_platform_id* get_opencl_platforms(cl_uint &platformnum)
{
    cl_int errnum;
    cl_platform_id *platformid;
    errnum = clGetPlatformIDs(0, NULL, &platformnum);
    if(errnum != CL_SUCCESS)
    {
        cerr << "unable to get OpenCL platforms\n";
        throw exception();
    }
    platformid = (cl_platform_id*)malloc(sizeof(cl_platform_id)*platformnum);
    errnum = clGetPlatformIDs(platformnum, platformid, NULL);
    if(errnum != CL_SUCCESS)
    {
        cerr << "unable to get OpenCL platforms\n";
        throw exception();
    }
    return platformid;
}

void* get_opencl_device_info(cl_device_id device, cl_device_info param_name, size_t &retsize)
{
    cl_int err;
    err = clGetDeviceInfo(device, param_name, 0, NULL, &retsize);
    if(err != CL_SUCCESS)
    {
        cerr << "unable to get initial OpenCL device info\n";
        throw exception();
    }
    void *info = malloc(retsize);
    err = clGetDeviceInfo(device, param_name, retsize, info, NULL);
    if(err != CL_SUCCESS)
    {
        cerr << "unable to get actual OpenCL device info\n";
        throw exception();
    }
    return info;
}

void print_mtx(matrice_data &mtx)
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

matrice_data matrice_data::operator* (const matrice_data& other)
{
    if(col != other.row)
        {
            std::cerr << "the condition of the if statement is fales in the operator Matrice::operator*\n";
            throw std::exception();
        }
    else
        {
            matrice_data mtx(row, other.col);
            float c = 0;
            for(int k = 0; k < row; k++)
                {
                    for(int l = 0; l < other.col; l++)
                        {
                            for(int i = 0; i < col; i++)
                                {
                                    c += data[k*col+i] * other[i][l];
                                }
                            mtx[k][l] = c;
                            c = 0;
                        }
                }
            return mtx;
        }
}

inline void matrice_data::destruct()
{
    if(this->data != NULL)
        {
            delete[] this->data;
            this->data = NULL;
        }
}

inline void matrice_data::equality(const matrice_data &mtx)
{
    row = mtx.row;
    col = mtx.col;
    try
       {
            data = new float[this->row * this->col];

       }
    catch(bad_alloc& ba)
        {
            cerr << "Matrice::copyconstructor: bad_alloc caught: " << ba.what() << endl;
            throw;
        }
    for(int i = 0; i < row*col; i++)
    {
        this->data[i] = mtx.data[i];
    }
}
matrice_data::matrice_data(const matrice_data& mtx)
{
    //this->destruct();
    this->equality(mtx);

}
matrice_data matrice_data::operator= (const matrice_data& mtx)
{
    if(this==&mtx) return *this;
    this->destruct();
    this->equality(mtx);
    return *this;

}

void conv(matrice_data &input, matrice_data &kernel, matrice_data &output, int stride=1)
{
    double helper;
    int r, c;
    r = c = 0;
    for(int i = kernel.row-1; i < input.row; i += stride)
        {
            for(int j = kernel.col-1; j < input.col; j += stride)
                {
                    helper = 0;
                    for(int k = kernel.row-1; k >= 0; k--)
                        {
                            for(int l = kernel.col-1; l >= 0; l--)
                                {
                                    helper += kernel[k][l] * input[i - k][j - l];
                                }
                        }
                    output[r][c] = helper;
                    c++;
                }
            r++;
            c = 0;
        }
}

int main(void)
{
    cl_uint platformnum;
    cl_uint numDevices;
    cl_context context;
    cl_int errorcode;
    cl_event event;
    cl_platform_id *platformid = get_opencl_platforms(platformnum);
    cl_context_properties properties [] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformid[0], 0};
    cl_device_id *deviceIds = get_opencl_devices(CL_DEVICE_TYPE_GPU, platformid[0], numDevices);
    context = clCreateContext(properties, numDevices, deviceIds, NULL, NULL, &errorcode);
    load_matrice_operations_programs(&context,deviceIds);
    matrice_operations op(&context, deviceIds);
    matrice_data A(5,5), B(5,5), C(5,5);
    for(int i = 0; i < A.row*A.col; i++)
    {
        A.data[i] = i;
    }
    for(int i = 0; i < B.row*B.col; i++)
    {
        B.data[i] = i;
    }
    for(int i = 0; i < C.row*C.col; i++)
    {
        C.data[i] = i;
    }
    A.create_opencl_buffer(context);
    B.create_opencl_buffer(context);
    C.create_opencl_buffer(context);

    /*op.add_matrices(A, B, C, 0, NULL, &event);
    errorcode = clEnqueueReadBuffer(op.command_queue, C.cl_mem_obj, CL_TRUE, 0, C.row*C.col * sizeof(float), C.data, 1, &event, NULL);
    print_mtx(C);
    op.transpose(A, B, 0, NULL, &event);
    errorcode = clEnqueueReadBuffer(op.command_queue, B.cl_mem_obj, CL_TRUE, 0, B.row*B.col * sizeof(float), B.data, 1, &event, NULL);
    print_mtx(B);
    op.multiply(A, B, C, 0, NULL, &event);
    errorcode = clEnqueueReadBuffer(op.command_queue, C.cl_mem_obj, CL_TRUE, 0, C.row*C.col * sizeof(float), C.data, 1, &event, NULL);*/
    op.hadamart(A, B, C, 0, NULL, &event);
    errorcode = clEnqueueReadBuffer(op.command_queue, C.cl_mem_obj, CL_TRUE, 0, C.row*C.col * sizeof(float), C.data, 1, &event, NULL);
    print_mtx(A);
    print_mtx(B);
    print_mtx(C);
    conv(A, B, C);
    print_mtx(C);
    // Clean up
    errorcode = clReleaseContext(context);
    return 0;

}
