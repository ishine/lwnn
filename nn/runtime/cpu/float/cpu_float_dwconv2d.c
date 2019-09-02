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
} layer_cpu_float_dwconv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void depthwise_convolve_HWC_ref_nonsquare(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const float * wt,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const float * bias,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y
		)
{
	float conv_out;
	int i_out_y, i_out_x, i_ch_out;
	int i_ker_y, i_ker_x;
	int in_row, in_col;

	for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
	{
		for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
		{
			for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
			{
				conv_out = bias[i_ch_out];
				for (i_ker_y = 0; i_ker_y < dim_kernel_y; i_ker_y++)
				{
					for (i_ker_x = 0; i_ker_x < dim_kernel_x; i_ker_x++)
					{
						in_row = stride_y * i_out_y + i_ker_y - padding_y;
						in_col = stride_x * i_out_x + i_ker_x - padding_x;
						if ((in_row >= 0) && (in_col >= 0) && (in_row < dim_im_in_y) && (in_col < dim_im_in_x))
						{
							conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + i_ch_out] *
								wt[(i_ker_y * dim_kernel_x + i_ker_x) * ch_im_out + i_ch_out];
						}
					}
				}
				Im_out[(i_out_y * dim_im_out_x + i_out_x) * ch_im_out + i_ch_out] = conv_out;
			}
		}
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_DWCONV2D_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_dwconv2d_context_t), sizeof(float));
}
int layer_cpu_float_DWCONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_dwconv2d_context_t* context = (layer_cpu_float_dwconv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float *weights = (float*)layer->blobs[0]->blob;
	float *bias = (float*)layer->blobs[1]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);


	ints = (int*)layer->blobs[0]->dims;
	knlY = ints[1];
	knlX = ints[2];

	ints = (int*)layer->blobs[2]->blob;
	padY = ints[0];
	padX = ints[1];
	strideY = ints[4];
	strideX = ints[5];

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d]\n",
			layer->name,
			knlY, knlX, padY, padX, strideY, strideX));

	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		depthwise_convolve_HWC_ref_nonsquare(IN+batch_sizeIn*batch,
			input_context->nhwc.W,
			input_context->nhwc.H,
			input_context->nhwc.C,
			weights,
			context->nhwc.C,
			knlX, knlY,
			padX, padY,
			strideX, strideY,
			bias,
			O+batch_sizeO*batch,
			context->nhwc.W,
			context->nhwc.H);
	}

	return r;
}
void layer_cpu_float_DWCONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
