/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
	cl_mem in;
} layer_cl_input_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	layer_cl_input_context_t* context;

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "input.cl", "input", NULL,
				sizeof(layer_cl_input_context_t));
	if(0 == r)
	{
		context = (layer_cl_input_context_t*)layer->C->context;
		context->in = NULL;
	}

	return r;
}

int layer_cl_INPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_input_context_t* context = (layer_cl_input_context_t*)layer->C->context;
	float* data;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	data = (float*) nn_get_input_data(nn, layer);
	if(NULL != data)
	{
		if(NULL != context->in)
		{
			clReleaseMemObject(context->in);
		}

		context->in = rte_cl_create_buffer(nn, NHWC_SIZE(context->nhwc), data);

		if(NULL != context->in)
		{
			r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 2,
						sizeof(cl_mem), &(context->in),
						sizeof(cl_mem), &(context->out[0]));
		}
		else
		{
			r = NN_E_NO_MEMORY;
		}
	}
	else
	{
		r = NN_E_NO_INPUT_BUFFER_PROVIDED;
	}

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
	}

	return r;
}

void layer_cl_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_input_context_t* context = (layer_cl_input_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(NULL != context->in)
		{
			clReleaseMemObject(context->in);
		}
		rte_cl_destory_layer_context(nn, layer);
	}
}
#endif /* DISABLE_RUNTIME_OPENCL */
