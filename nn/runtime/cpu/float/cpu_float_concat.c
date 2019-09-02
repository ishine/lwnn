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
} layer_cpu_float_concat_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_CONCAT_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_concat_context_t), sizeof(float));
}
int layer_cpu_float_CONCAT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_concat_context_t* context = (layer_cpu_float_concat_context_t*)layer->C->context;
	int axis = RTE_FETCH_INT32(layer->blobs[0]->blob, 0);
	const layer_t** input = layer->inputs;
	layer_cpu_context_t* input_context;

	float* pin;
	float* pout = (float*)context->out[0];
	size_t n_block;
	size_t in_stride;
	size_t out_stride;
	size_t i,j;

	n_block = 1;
	for (i = 0; i < axis; i++)
	{	/* Calculate the number of block to concat. (the other shapes before the concat axis) */
		n_block *= RTE_FETCH_INT32(&(context->nhwc), i);
	}
	out_stride = 1;
	for(j = axis; j <= 3; j++)
	{
		out_stride *= RTE_FETCH_INT32(&(context->nhwc), j);
	}

	NNLOG(NN_DEBUG, ("execute %s: axis=%d, n_block=%d, out stride=%d\n", layer->name, axis, n_block, out_stride));

	while((*input) != NULL)
	{	/* concat all input layers */
		input_context = (layer_cpu_context_t*)(*input)->C->context;
		pin = (float*)input_context->out[0];

		in_stride = 1;
		for(j = axis; j <= 3; j++)
		{
			in_stride *= RTE_FETCH_INT32(&(input_context->nhwc), j);
		}

		NNLOG(NN_DEBUG, ("concat %s, in stride=%d\n", (*input)->name, in_stride));
		for(i=0; i<n_block; i++)
		{
			memcpy(pout+i*out_stride, pin, in_stride*sizeof(float));
			pin += in_stride;
		}
		pout += in_stride;
		input++;
	}

	return r;
}
void layer_cpu_float_CONCAT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
