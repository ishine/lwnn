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
} layer_cl_pooling_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int layer_cl_pooling_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	const char* kernel;

	switch(layer->op)
	{
		case L_OP_MAXPOOL:
			kernel = "maxpool";
			break;
		default:
			assert(0);
			break;
	}

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "pooling.cl", kernel,
				sizeof(layer_cl_pooling_context_t));

	return r;
}

static int layer_cl_pooling_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_pooling_context_t* context = (layer_cl_pooling_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	ints = (int*)layer->blobs[0]->blob;
	knlY = ints[0];
	knlX = ints[1];
	padY = ints[2];
	padX = ints[3];
	strideY = ints[4];
	strideX = ints[5];

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHC, 10,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &(input_context->nhwc.W),
					sizeof(int), &(input_context->nhwc.H),
					sizeof(int), &knlX,
					sizeof(int), &knlY,
					sizeof(int), &padX,
					sizeof(int), &padY,
					sizeof(int), &strideX,
					sizeof(int), &strideY);

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
	}

	return r;
}

static void layer_cl_pooling_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_MAXPOOL_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_init(nn, layer);
}

int layer_cl_MAXPOOL_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_execute(nn, layer);
}

void layer_cl_MAXPOOL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_pooling_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */