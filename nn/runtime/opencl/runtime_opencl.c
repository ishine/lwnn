/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include <CL/cl.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	cl_context context;
	cl_device_id device;
	cl_command_queue command_queue;

} runtime_opencl_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static cl_context cl_create_context()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		NN_LOG(NN_ERROR, ("Failed to find any OpenCL platforms.\n"));
	}
	else
	{
		cl_context_properties contextProperties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)firstPlatformId,
			0
		};
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	}

	return context;
}

static cl_command_queue cl_create_command_queue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

	if (deviceBufferSize <= 0)
	{
		NN_LOG(NN_ERROR, ("No OpenCL devices available."));
	}
	else
	{
		devices = malloc(deviceBufferSize);
		errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

		commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

		*device = devices[0];
		free(devices);
	}

	return commandQueue;
}

static int cl_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	NN_LOG(NN_DEBUG, (" CL run %-16s: op=%d\n", layer->name, layer->op));
	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t runtime_opencl_create(const nn_t* nn)
{
	runtime_opencl_t* rt = NULL;

	rt = malloc(sizeof(runtime_opencl_t));
	if(NULL != rt)
	{
		rt->context = cl_create_context();
	}

	if(NULL == rt->context)
	{
		free(rt);
		rt = NULL;
	}
	else
	{
		rt->command_queue = cl_create_command_queue(rt->context, &rt->device);
	}

	if(NULL == rt->command_queue)
	{
		free(rt->context);
		free(rt);
		rt = NULL;
	}

	return rt;
}

int runtime_opencl_execute(const nn_t* nn)
{
	int r = 0;

	const layer_t* const* network;
	const layer_t* layer;

	network = nn->network;

	layer = *network++;
	while((NULL != layer) && (0 == r))
	{
		r = cl_execute_layer(nn, layer);
		layer = *network++;
	}

	return r;
}