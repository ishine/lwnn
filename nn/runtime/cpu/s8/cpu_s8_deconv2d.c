/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_S8
#include "../runtime_cpu.h"
#include "algorithm.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_S8_CONTEXT_MEMBER;
} layer_cpu_s8_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void deconvolve_HWC_ref_nonsquare(const int8_t * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const int8_t * wt,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const int32_t * bias,
		int8_t * Im_out,
		const int32_t *output_shift,
		const int32_t *output_mult,
		const int32_t out_offset,
		const int32_t input_offset,
		const int32_t out_activation_min,
		const int32_t out_activation_max,
		const int dim_im_out_x,
		const int dim_im_out_y
		)
{
	int i, j, k, l, m, n;
	int conv_out;
	int in_row, in_col;
	int kernel_start_x,kernel_end_x;
	int kernel_start_y,kernel_end_y;
	int in_row_start, in_col_start;
	int is_zero;

	for (i = 0; i < ch_im_out; i++) {
		for (j = 0; j < dim_im_out_y; j++) {
			is_zero = alg_deconv2d_calculate_position(j, stride_y, padding_y, dim_kernel_y,
					dim_im_in_y, &in_row_start, &kernel_start_y, &kernel_end_y);

			if(is_zero) {
				conv_out = bias[i];
				conv_out = arm_nn_requantize(conv_out, output_mult[i], output_shift[i]);
				conv_out += out_offset;
				conv_out = MAX(conv_out, out_activation_min);
				conv_out = MIN(conv_out, out_activation_max);
				for (k = 0; k < dim_im_out_x; k++) {
					Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)conv_out;
				}
				continue;
			}

			for (k = 0; k < dim_im_out_x; k++) {
				conv_out = bias[i];

				is_zero = alg_deconv2d_calculate_position(k, stride_x, padding_x, dim_kernel_x,
						dim_im_in_x, &in_col_start, &kernel_start_x, &kernel_end_x);

				if(is_zero) {
					conv_out = arm_nn_requantize(conv_out, output_mult[i], output_shift[i]);
					conv_out += out_offset;
					conv_out = MAX(conv_out, out_activation_min);
					conv_out = MIN(conv_out, out_activation_max);
					Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)conv_out;
					continue;
				}

				for (m = kernel_start_y, in_row = in_row_start; m <= kernel_end_y; m+=stride_y, in_row++) {
					for (n = kernel_start_x, in_col = in_col_start; n <= kernel_end_x; n+=stride_x, in_col++) {
						if ((in_row >= 0) && (in_col >= 0) &&
							(in_row < dim_im_in_y) && (in_col < dim_im_in_x)) {
							for (l = 0; l < ch_im_in; l++) {
								conv_out += (Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] + input_offset) *
									wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in + l];
							}
						}
					}
				}

				conv_out = arm_nn_requantize(conv_out, output_mult[i], output_shift[i]);
				conv_out += out_offset;
				conv_out = MAX(conv_out, out_activation_min);
				conv_out = MIN(conv_out, out_activation_max);
				Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)conv_out;
			}
		}
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_s8_DECONV2D_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_s8_conv2d_context_t), sizeof(int8_t));
}

int layer_cpu_s8_DECONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_conv2d_context_t* context = (layer_cpu_s8_conv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_s8_context_t* input_context = (layer_cpu_s8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *weights = (int8_t*)layer->blobs[1]->blob;
	int32_t *bias = (int32_t*)layer->blobs[2]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;
	int omin;

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	ints = (int*)layer->blobs[1]->dims;
	knlY = ints[1];
	knlX = ints[2];

	ints = (int*)layer->blobs[3]->blob;
	strideY = ints[4];
	strideX = ints[5];

	padY = alg_deconv2d_calculate_padding(knlY, strideY, input_context->nhwc.H, context->nhwc.H);
	padX = alg_deconv2d_calculate_padding(knlX, strideX, input_context->nhwc.W, context->nhwc.W);

	omin = ints[6];

	NNLOG(NN_DEBUG, (" kernel=[%d %d], pads=[%d %d], strides=[%d %d], Z=%d, %d -> %d\n",
			knlY, knlX, padY, padX, strideY, strideX,
			LAYER_Z(layer), LAYER_Q(input), LAYER_Q(layer)));

	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		deconvolve_HWC_ref_nonsquare(IN+batch_sizeIn*batch,
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
					(const int32_t*)layer->blobs[5]->blob,
					(const int32_t*)layer->blobs[4]->blob,
					-LAYER_Z(layer),
					LAYER_Z(input),
					omin,
					INT8_MAX,
					context->nhwc.W,
					context->nhwc.H
					);
	}
	return r;
}

void layer_cpu_s8_DECONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_S8 */
