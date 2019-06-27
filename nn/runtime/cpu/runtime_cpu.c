/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int cpu_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	NN_LOG(NN_DEBUG, (" CPU run %-16s: op=%d\n", layer->name, layer->op));
	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t runtime_cpu_create(const nn_t* nn)
{
	runtime_t rt = (void*)1;

	return rt;
}

int runtime_cpu_execute(const nn_t* nn)
{
	return runtime_execute_helper(nn, cpu_execute_layer);
}
