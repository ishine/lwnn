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
} layer_cl_upsample_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_UPSAMPLE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_create_layer_common(nn, layer,
			OPENCL_PATH "upsample.cl", "upsample2d", NULL,
			sizeof(layer_cl_upsample_context_t));
}
int layer_cl_UPSAMPLE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_upsample_context_t* context = (layer_cl_upsample_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	int strideX, strideY;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	strideY = context->nhwc.H/input_context->nhwc.H;
	strideX = context->nhwc.W/input_context->nhwc.W;

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_C, 4,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &strideX,
					sizeof(int), &strideY);

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, FALSE, NULL);
	}

	return r;
}
void layer_cl_UPSAMPLE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
