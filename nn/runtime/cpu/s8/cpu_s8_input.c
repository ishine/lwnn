/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_S8
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_S8_CONTEXT_MEMBER;
} layer_cpu_s8_input_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_s8_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_input_context_t* context;

	r = rte_cpu_create_layer_context(nn, layer,
				sizeof(layer_cpu_s8_input_context_t), 1);

	if(0 == r)
	{
		context = (layer_cpu_s8_input_context_t*)layer->C->context;

		RTE_CPU_LOG_LAYER_SHAPE(layer);

		context->out[0] = NULL;
	}

	return r;
}

int layer_cpu_s8_INPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_input_context_t* context = (layer_cpu_s8_input_context_t*)layer->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	context->out[0] = nn_get_input_data(nn, layer);

	if(NULL == context->out[0])
	{
		r = NN_E_NO_INPUT_BUFFER_PROVIDED;
	}

	return r;
}

void layer_cpu_s8_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_S8 */