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
int nn_log_level = NN_INFO;
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(const network_t* network, runtime_type_t runtime_type)
{
	nn_t* nn;

	nn = malloc(sizeof(nn_t));
	if(NULL != nn)
	{
		nn->runtime_type = runtime_type;
		nn->network = network;
		#if !defined(DISABLE_NN_SCRATCH) || \
			!defined(DISABLE_RTE_FALLBACK)
		nn->scratch.size = 0;
		nn->scratch.area = NULL;
		#endif

		nn->runtime = rte_create(nn);
	}

	if(NULL != nn->runtime)
	{
		int r = rte_init(nn);

		#ifndef DISABLE_NN_SCRATCH
		if((0 == r) && (0 != nn->scratch.size))
		{
			nn->scratch.area = malloc(nn->scratch.size);

			if(NULL == nn->scratch.area)
			{
				r = NN_E_NO_MEMORY;
			}
		}
		#endif
		if(0 != r)
		{
			NNLOG(NN_ERROR,("nn create failed with %d\n", r));
			rte_destory(nn);
			nn->runtime = NULL;
		}
	}

	if(NULL == nn->runtime)
	{
		free(nn);
		nn = NULL;
	}

	return nn;
}

void nn_set_log_level(int level)
{
	nn_log_level = level;
}

int nn_predict(nn_t* nn)
{
	return rte_execute(nn);
}

void* nn_get_input_data(const nn_t* nn, const layer_t* layer)
{
	void* data = NULL;

	const nn_input_t* const* input = nn->network->inputs;

	while(((*input) != NULL) && (NULL == data))
	{
		if((*input)->layer == layer)
		{
			data = (*input)->data;
		}

		input++;
	}

	return data;
}

void* nn_get_output_data(const nn_t* nn, const layer_t* layer)
{
	void* data = NULL;

	const nn_output_t* const* output = nn->network->outputs;

	while(((*output) != NULL) && (NULL == data))
	{
		if((*output)->layer == layer)
		{
			data = (*output)->data;
		}

		output++;
	}

	return data;
}

void nn_destory(nn_t* nn)
{
	if(NULL != nn)
	{
		rte_destory(nn);
		#ifndef DISABLE_NN_SCRATCH
		if(NULL != nn->scratch.area)
		{
			free(nn->scratch.area);
		}
		#endif
		free(nn);
	}
}

#if !defined(DISABLE_NN_SCRATCH) || \
	!defined(DISABLE_RTE_FALLBACK)
void nn_request_scratch(const nn_t* nn, size_t sz)
{
	nn_t * pnn = (nn_t*)nn;
	if(sz > pnn->scratch.size)
	{
		pnn->scratch.size = sz;
	}
}
#endif

void* nn_allocate_input(const layer_t* layer)
{
	void* mem;
	layer_data_type_t dtype;
	size_t sz = layer_get_size(layer);

	if(NULL != layer->C->context)
	{
		dtype = layer->C->context->dtype;
	}
	else
	{
		dtype = layer->dtype;
	}

	switch(dtype)
	{
		case L_DT_INT8:
		case L_DT_UINT8:
			break;
		case L_DT_INT16:
		case L_DT_UINT16:
			sz *= 2;
			break;
		case L_DT_INT32:
		case L_DT_UINT32:
		case L_DT_FLOAT:
			sz *= 4;
			break;
		default:
			NNLOG(NN_ERROR,("invalid dtype(%d) for %s\n",
					dtype, layer->name));
			sz = 0;
			break;
	}

	if(sz > 0)
	{
		mem = malloc(sz);
	}

	return mem;
}

void* nn_allocate_output(const layer_t* layer)
{
	return nn_allocate_input(layer);
}

void nn_free_input(void* input)
{
	free(input);
}

void nn_free_output(void* output)
{
	free(output);
}

#ifdef L_BLOB_NOT_BUILTIN
int nn_load(const network_t* network, nn_blob_loader_t loader, void* provider)
{
	int r = 0;
	const layer_t* const* layers;
	const layer_t* layer;
	const layer_blob_t* const* blobs;
	const layer_blob_t* blob;
	size_t size;
	const int* dims;

	layers = network->layers;
	layer = *layers++;
	while((NULL != layer) && (0 == r)) {
		blobs = layer->blobs;
		if(blobs != NULL) {
			blob = *blobs++;
			while((NULL != blob) && (0 == r)) {
				dims = blob->dims;
				size = *dims++;
				while(*dims != 0) {
					size *= *dims++;
				}
				switch(blob->dtype) {
					case L_DT_INT16:
					case L_DT_UINT16:
						size *= sizeof(int16_t);
						break;
					case L_DT_INT32:
					case L_DT_UINT32:
						size *= sizeof(int32_t);
						break;
					case L_DT_FLOAT:
						size *= sizeof(float);
						break;
					default:
						break;
				}
				r = loader(provider, (void*)blob->blob, size);
				blob = *blobs++;
			}
		}
		layer = *layers++;
	}

	if(0 == r) { /* check all weights are consumed */
		r = loader(provider, (void*)&size, 1);
		if(r != 0) {
			r = 0;
		} else {
			r = NN_E_INVALID_WEIGHTS_LOADER;
			NNLOG(NN_ERROR, ("mismatched weights for %s\n", network->name));
		}
	}
	return r;
}
#endif
