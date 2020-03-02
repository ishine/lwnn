/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void layer_cpu_float_max(float* A, float* B, float* O, size_t sz)
{
	size_t i;
	for(i=0; i<sz; i++)
	{
		if(A[i] > B[i])
		{
			O[i] = A[i];
		}
		else
		{
			O[i] = B[i];
		}
	}
}

static void layer_cpu_float_min(float* A, float* B, float* O, size_t sz)
{
	size_t i;
	for(i=0; i<sz; i++)
	{
		if(A[i] < B[i])
		{
			O[i] = A[i];
		}
		else
		{
			O[i] = B[i];
		}
	}
}

static void layer_cpu_float_add(float* A, float* B, float* O, size_t sz)
{
	size_t i;
	for(i=0; i<sz; i++)
	{
		O[i] = A[i] + B[i];
	}
}
static int layer_cpu_float_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_eltwise_context_t), sizeof(float));
}

static int layer_cpu_float_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_eltwise_context_t* context = (layer_cpu_float_eltwise_context_t*)layer->C->context;
	const layer_t* inputA = layer->inputs[0];
	const layer_t* inputB = layer->inputs[1];
	layer_cpu_context_t* inputA_context;
	layer_cpu_context_t* inputB_context;
	size_t sz = NHWC_SIZE(context->nhwc);
	float* A;
	float* B;
	float* O;

	inputA_context = (layer_cpu_context_t*)inputA->C->context;
	inputB_context = (layer_cpu_context_t*)inputB->C->context;

	A = (float*)inputA_context->out[0];
	B = (float*)inputB_context->out[0];
	O = (float*)context->out[0];

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	switch(layer->op)
	{
		case L_OP_MAXIMUM:
			layer_cpu_float_max(A, B, O, sz);
			break;
		case L_OP_ADD:
			layer_cpu_float_add(A, B, O, sz);
			break;
		case L_OP_MINIMUM:
			layer_cpu_float_min(A, B, O, sz);
			break;
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}

	return r;
}

static void layer_cpu_float_eltwise_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}

int layer_cpu_float_MINIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_MINIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_MINIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}

int layer_cpu_float_ADD_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_ADD_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_ADD_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
