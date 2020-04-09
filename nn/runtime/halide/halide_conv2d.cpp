/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "runtime_halide.h"
#include "GenOps/Convolution_generator.cpp"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef int (*HL_Convolution_t)(struct halide_buffer_t *, struct halide_buffer_t *,
				struct halide_buffer_t *, int32_t, int32_t, int32_t, int32_t,
				int32_t, struct halide_buffer_t *);
typedef struct {
	LAYER_HALIDE_CONTEXT_MEMBER;
	Halide::Buffer<float>* W;
	Halide::Buffer<float>* B;
	ConvolutionLayer* L;
	void* dll;
	HL_Convolution_t HL_Convolution;
} layer_halide_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_halide_CONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_halide_conv2d_context_t* context;
	layer_halide_context_t* input_context;

	r = rte_halide_create_layer_common(nn, layer, sizeof(layer_halide_conv2d_context_t));

	if(0 == r) {
		input_context = (layer_halide_context_t*)layer->inputs[0]->C->context;
		context = (layer_halide_conv2d_context_t*) layer->C->context;
		context->dll = NULL;
		context->L = NULL;
		context->W = rte_halide_create_buffer_from_blob(nn, layer->blobs[0]);
		if(NULL != context->W) {
			context->B = rte_halide_create_buffer_from_blob(nn, layer->blobs[1]);
			if(NULL == context->B) {
				delete context->W;
				context->W = NULL;
				r = NN_E_NO_MEMORY;
			}
		} else {
			context->B = NULL;
			r = NN_E_NO_MEMORY;
		}
	}

	if(0 == r) {
		context->HL_Convolution = (HL_Convolution_t)halide_load_algorithm("HL_Convolution", &context->dll);
	}

	if((0 == r) && (NULL == context->HL_Convolution)) {
		context->L = new ConvolutionLayer();
		if(NULL == context->L) {
			delete context->W;
			delete context->B;
			context->W = NULL;
			context->B = NULL;
			r = NN_E_NO_MEMORY;
		} else {
			Halide::Var n, y, x, c;
			ConvolutionLayer& L = *context->L;
			Halide::Buffer<float>& in = *(Halide::Buffer<float>*)input_context->out[0];
			Halide::Buffer<float>& W = *(context->W);
			Halide::Buffer<float>& B = *(context->B);
			W.set_name(std::string(layer->name)+"_W");
			B.set_name(std::string(layer->name)+"_B");

			int* ints = (int*)layer->blobs[2]->blob;
			L.knlY = layer->blobs[0]->dims[1];
			L.knlX = layer->blobs[0]->dims[2];
			L.padY = ints[0];
			L.padX = ints[1];
			L.strideY = ints[4];
			L.strideX = ints[5];
			L.iH = input_context->nhwc.H;
			L.iW = input_context->nhwc.W;
			L.iC = input_context->nhwc.C;

			L.W(c,x,y,n) = W(c,x,y,n);
			L.B(c) = B(c);
			L.input(c,x,y,n) = in(c,x,y,n);
			L.generate();
		}
	}

	return r;
}

int layer_halide_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_halide_conv2d_context_t* context = (layer_halide_conv2d_context_t*)layer->C->context;
	Halide::Buffer<float>& out = *(Halide::Buffer<float>*)context->out[0];
	if(NULL != context->HL_Convolution) {
		layer_halide_context_t* input_context = (layer_halide_context_t*)layer->inputs[0]->C->context;
		Halide::Buffer<float>& in = *(Halide::Buffer<float>*)input_context->out[0];
		Halide::Buffer<float>& W = *(context->W);
		Halide::Buffer<float>& B = *(context->B);
		Halide::Runtime::Buffer<float> rin(in.data(), {input_context->nhwc.C, input_context->nhwc.W, input_context->nhwc.H, input_context->nhwc.N});
		Halide::Runtime::Buffer<float> rW(W.data(), {layer->blobs[0]->dims[3], layer->blobs[0]->dims[2], layer->blobs[0]->dims[1], layer->blobs[0]->dims[0]});
		Halide::Runtime::Buffer<float> rB(B.data(), {layer->blobs[1]->dims[0]});
		Halide::Runtime::Buffer<float> rout(out.data(), {context->nhwc.C, context->nhwc.W, context->nhwc.H, context->nhwc.N});
		int* ints = (int*)layer->blobs[2]->blob;
		int padY = ints[0];
		int padX = ints[1];
		int strideY = ints[4];
		int strideX = ints[5];
		int iC = input_context->nhwc.C;

		r = context->HL_Convolution(rin, rW, rB, strideX, strideY, padX, padY, iC, rout);
	} else {
		ConvolutionLayer& L = *context->L;
		out = L.conv.realize(context->nhwc.C, context->nhwc.W, context->nhwc.H, context->nhwc.N);
	}
	return r;
}

void layer_halide_CONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_halide_conv2d_context_t* context = (layer_halide_conv2d_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(context->W) delete context->W;
		if(context->B) delete context->B;
		if(context->L) delete context->L;
		if(context->dll) dlclose(context->dll);
		rte_halide_destory_layer_context(nn, layer);
	}
}
