#include "opencl_setup.h"
#include <iostream>
#include <fstream>

using namespace std;

OpenclSetup::OpenclSetup()
{
    cl_int errorcode;
    this->platformid = this->get_opencl_platforms(this->platformnum);
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformid[0], 0, 0};
    this->deviceIds = get_opencl_devices(CL_DEVICE_TYPE_GPU, this->platformid[0], this->numDevices);
    this->context = clCreateContext(properties, this->numDevices, this->deviceIds, NULL, NULL, &errorcode);
    if(errorcode != CL_SUCCESS)
    {
        cerr << "unable to create OpenCL context\n";
        throw exception();
    }
}

OpenclSetup::~OpenclSetup()
{
    ;
}

void OpenclSetup::print_opencl_platform_info(cl_platform_id &platformid, cl_platform_info name)
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

cl_device_id* OpenclSetup::get_opencl_devices(cl_device_type device_type, cl_platform_id &platform, cl_uint &numDevices)
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

cl_platform_id* OpenclSetup::get_opencl_platforms(cl_uint &platformnum)
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

void* OpenclSetup::get_opencl_device_info(cl_device_id device, cl_device_info param_name, size_t &retsize)
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
            size_t log_size;
            clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char *) malloc(log_size);
            clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            cout << log << endl;

            throw exception();
        }
    return program;
}
