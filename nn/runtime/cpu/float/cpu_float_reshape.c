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
} layer_cpu_float_reshape_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_RESHAPE_init(const nn_t* nn, const layer_t* layer)
{
	int r =0;
	layer_cpu_float_reshape_context_t* context;

	const layer_t* input;
	layer_cpu_context_t* input_context;

	r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_float_reshape_context_t), 1);

	if(0 == r)
	{
		context = (layer_cpu_float_reshape_context_t*)layer->C->context;

		input = layer->inputs[0];
		input_context = (layer_cpu_context_t*)input->C->context;

		if(NULL != input_context->out[0])
		{
			/* reuse its input layer's output buffer */
			rte_cpu_take_buffer(input_context->out[0], layer);
		}
	}

	return r;
}

int layer_cpu_float_RESHAPE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_reshape_context_t* context = (layer_cpu_float_reshape_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;

	float* IN = (float*)input_context->out[0];

#ifndef DISABLE_DYNAMIC_SHAPE
	rte_cpu_dynamic_reshape(layer, input_context);
#endif

	NNLOG(NN_DEBUG, ("execute %s: %d %d %d %d\n", layer->name, L_SHAPES(layer)));

	context->out[0] = IN;	/* yes, just set up the output */


	return r;
}

void layer_cpu_float_RESHAPE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
