/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
	alg_broadcast_t broadcast;
	layer_context_t* inputA_context;
	layer_context_t* inputB_context;
} layer_cl_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int layer_cl_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	const char* kernel;
	layer_cl_eltwise_context_t* context;
	alg_broadcast_t broadcast = ALG_BROADCAST_NONE;
	layer_context_t* inputA_context = (layer_context_t*)layer->inputs[0]->C->context;
	layer_context_t* inputB_context = (layer_context_t*)layer->inputs[1]->C->context;
	r = alg_broadcast_prepare(&inputA_context, &inputB_context, &broadcast);

	if(0 == r) {
	switch(layer->op+broadcast)
	{
		case L_OP_MAXIMUM:
			kernel = "maximum";
			break;
		case L_OP_MAXIMUM+ALG_BROADCAST_ONE:
			kernel = "maximum_broadcast_one";
			break;
		case L_OP_MAXIMUM+ALG_BROADCAST_CHANNEL:
			kernel = "maximum_broadcast_channel";
			break;
		case L_OP_ADD:
			kernel = "add";
			break;
		case L_OP_ADD+ALG_BROADCAST_ONE:
			kernel = "add_broadcast_one";
			break;
		case L_OP_ADD+ALG_BROADCAST_CHANNEL:
			kernel = "add_broadcast_channel";
			break;
		case L_OP_MINIMUM:
			kernel = "minimum";
			break;
		case L_OP_MINIMUM+ALG_BROADCAST_ONE:
			kernel = "minimum_broadcast_one";
			break;
		case L_OP_MINIMUM+ALG_BROADCAST_CHANNEL:
			kernel = "minimum_broadcast_channel";
			break;
		case L_OP_MUL:
			kernel = "mul";
			break;
		case L_OP_MUL+ALG_BROADCAST_ONE:
			kernel = "mul_broadcast_one";
			break;
		case L_OP_MUL+ALG_BROADCAST_CHANNEL:
			kernel = "mul_broadcast_channel";
			break;
		default:
			assert(0);
			break;
	}

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "eltwise.cl", kernel, NULL,
				sizeof(layer_cl_eltwise_context_t));
	}

	if(0 == r) {
		context = (layer_cl_eltwise_context_t*)layer->C->context;
		context->broadcast = broadcast;
		context->inputA_context = inputA_context;
		context->inputB_context = inputB_context;
	}

	return r;
}

static int layer_cl_eltwise_set_args(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_eltwise_context_t* context = (layer_cl_eltwise_context_t*)layer->C->context;
	alg_broadcast_t broadcast = context->broadcast;
	layer_cl_context_t* inputA_context = (layer_cl_context_t*)context->inputA_context;
	layer_cl_context_t* inputB_context = (layer_cl_context_t*)context->inputB_context;

	if(ALG_BROADCAST_CHANNEL == broadcast) {
		int nC4 = (context->nhwc.C+3)>>2;
		r = rte_cl_set_layer_args(nn, layer, 0, 4,
					sizeof(cl_mem), &(inputA_context->out[0]),
					sizeof(cl_mem), &(inputB_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &nC4);
	} else {
		r = rte_cl_set_layer_args(nn, layer, 0, 3,
					sizeof(cl_mem), &(inputA_context->out[0]),
					sizeof(cl_mem), &(inputB_context->out[0]),
					sizeof(cl_mem), &(context->out[0]));
	}
	return r;
}

static int layer_cl_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, FALSE, NULL);
}

static void layer_cl_eltwise_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_MAXIMUM_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_set_args(nn, layer);
}

int layer_cl_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}

int layer_cl_MINIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_MINIMUM_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_set_args(nn, layer);
}

int layer_cl_MINIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_MINIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}

int layer_cl_ADD_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_ADD_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_set_args(nn, layer);
}

int layer_cl_ADD_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_ADD_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}

int layer_cl_MUL_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_MUL_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_set_args(nn, layer);
}

int layer_cl_MUL_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_MUL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */
