#ifndef OPENCL_SETUP_H_INCLUDED
#define OPENCL_SETUP_H_INCLUDED

#include <CL/cl.h>


class OpenclSetup{
    public:
    cl_uint platformnum;
    cl_uint numDevices;
    cl_context context;
    cl_event event;
    cl_platform_id *platformid;// = get_opencl_platforms(platformnum);
    //cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformid[0], 0};
    cl_device_id *deviceIds;// = get_opencl_devices(CL_DEVICE_TYPE_GPU, platformid[0], numDevices);
    //context = clCreateContext(properties, numDevices, deviceIds, NULL, NULL, &errorcode);
    OpenclSetup();
    ~OpenclSetup();
    void print_opencl_platform_info(cl_platform_id &platformid, cl_platform_info name);
    cl_device_id* get_opencl_devices(cl_device_type device_type, cl_platform_id &platform, cl_uint &numDevices);
    cl_platform_id* get_opencl_platforms(cl_uint &platformnum);
    void* get_opencl_device_info(cl_device_id device, cl_device_info param_name, size_t &retsize);
};


#endif // OPENCL_SETUP_H_INCLUDED
