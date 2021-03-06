# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn.core import *
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np

__all__ = ['caffe2lwnn']

class CaffeConverter():
    def __init__(self, caffe_model, caffe_weights):
        self.net = caffe_pb2.NetParameter()
        text_format.Merge(open(caffe_model,'r').read(),self.net)
        if(caffe_weights is None):
            self.caffe_model = caffe.Net(caffe_model,caffe.TEST)
        else:
            self.caffe_model = caffe.Net(caffe_model,caffe_weights,caffe.TEST)
        if(len(self.net.layer)==0):    #some prototxts use "layer", some use "layers"
            self.layers = self.net.layers
        else:
            self.layers = self.net.layer
        self.TRANSLATOR = {
            'Convolution': self.to_LayerConv,
            'Deconvolution': self.to_LayerDeconvolution,
            'Permute': self.to_LayerTranspose,
            'PriorBox': self.to_LayerPriorBox,
            'DetectionOutput': self.to_LayerDetectionOutput,
            'Concat': self.to_LayerConcat,
            'Softmax': self.to_LayerSoftmax,
            'Pooling': self.to_LayerPooling,
            'BN': self.to_LayerBN,
            'Normalize': self.to_LayerNormalize,
            'Eltwise': self.to_LayerEltwise,
            'PReLU': self.to_LayerPReLU,
             }
        self.opMap = { 
            'ReLU': 'Relu',
            'PReLU': 'PRelu',
            'Flatten': 'Reshape',
            'BN':'BatchNormalization',
            'Deconvolution':'ConvTranspose',
            }
        self.convert()

    @property
    def model(self):
        return self.lwnn_model

    def to_LayerCommon(self, cly, op=None):
        name = str(cly.name)
        blob = self.caffe_model.blobs[cly.top[0]]
        layer = LWNNLayer(name=name,
                  outputs=[str(o) for o in cly.top],
                  inputs=[str(o) for o in cly.bottom],
                  shape=list(blob.data.shape))
        if(op == None):
            op = str(cly.type)
        if(op in self.opMap):
            op = self.opMap[op]
        layer['op'] = op 
        return layer

    def to_LayerInput(self, cly):
        layer = self.to_LayerCommon(cly, 'Input')
        return layer

    def get_field(self, param, field, default):
        if('%s:'%(field) in str(param)):
            v = param.__getattribute__(field)
        else:
            v = default
        return v

    def get_field_hw(self, param, sname, hname, wname, default):
        s = self.get_field(param, sname, None)
        if(s!=None):
            try:
                s = s[0]
            except:
                pass
            v = [s,s]
        else:
            h = self.get_field(param, hname, default[0])
            w = self.get_field(param, wname, default[1])
            try:
                h = h[0]
                w = w[0]
            except:
                pass
            v = [h,w]
        return v

    def to_LayerConv(self, cly):
        layer = self.to_LayerCommon(cly, 'Conv')
        name = layer['name']
        params = self.caffe_model.params[name]
        layer['weights'] = params[0].data
        if(len(params) > 1):
            layer['bias'] = params[1].data
        else:
            num_output = layer['weights'].shape[0]
            layer['bias'] = np.zeros((num_output), np.float32)
        padW,padH = self.get_field_hw(cly.convolution_param, 'pad', 'pad_h', 'pad_w', [0,0])
        layer['pads'] = [padW,padH,0,0]
        layer['strides'] = self.get_field_hw(cly.convolution_param, 'stride', 'stride_h', 'stride_w', [1,1])
        layer['group'] = self.get_field(cly.convolution_param, 'group', 1)
        dilation = self.get_field(cly.convolution_param, 'dilation', None)
        if(dilation != None):
            if(len(dilation) == 1):
                dilation = dilation[0]
                layer['dilations'] = [dilation, dilation]
            else:
                 layer['dilations'] = dilation
        return layer

    def to_LayerDeconvolution(self, cly):
        layer = self.to_LayerCommon(cly)
        name = layer['name']
        params = self.caffe_model.params[name]
        layer['weights'] = params[0].data
        layer['bias'] = params[1].data
        padW,padH = self.get_field_hw(cly.convolution_param, 'pad', 'pad_h', 'pad_w', [0,0])
        layer['pads'] = [padW,padH,0,0]
        layer['strides'] = self.get_field_hw(cly.convolution_param, 'stride', 'stride_h', 'stride_w', [1,1])
        layer['group'] = self.get_field(cly.convolution_param, 'group', 1)
        return layer

    def to_LayerTranspose(self, cly):
        layer = self.to_LayerCommon(cly, 'Transpose')
        layer['perm'] = [d for d in cly.permute_param.order]
        return layer

    def to_LayerPriorBox(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['min_size'] = cly.prior_box_param.min_size
        layer['max_size'] = self.get_field(cly.prior_box_param, 'max_size', [])
        layer['aspect_ratio'] = cly.prior_box_param.aspect_ratio
        layer['variance'] = [v for v in cly.prior_box_param.variance]
        layer['offset'] = cly.prior_box_param.offset
        layer['flip'] = cly.prior_box_param.flip
        layer['clip'] = cly.prior_box_param.clip
        return layer

    def to_LayerDetectionOutput(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['num_classes'] = cly.detection_output_param.num_classes
        layer['nms_threshold'] = cly.detection_output_param.nms_param.nms_threshold
        layer['top_k'] = self.get_field(cly.detection_output_param.nms_param, 'top_k', -1)
        layer['code_type'] = cly.detection_output_param.code_type
        layer['keep_top_k'] = cly.detection_output_param.keep_top_k
        layer['confidence_threshold'] = self.get_field(cly.detection_output_param, 'confidence_threshold', float('inf'))# default -FLT_MAX
        layer['share_location'] = cly.detection_output_param.share_location
        layer['background_label_id'] = cly.detection_output_param.background_label_id
        shape = layer['shape']
        num_kept = cly.detection_output_param.num_classes*2 if (shape[2]==1) else shape[2]
        if(num_kept < 100):
            num_kept = 100
        layer['shape'] = [shape[0],
                          shape[3],
                          num_kept,
                          shape[1]]
        return layer

    def to_LayerConcat(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['axis'] = cly.concat_param.axis
        return layer

    def to_LayerSoftmax(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['axis'] = cly.softmax_param.axis
        return layer

    def to_LayerPooling(self, cly):
        layer = self.to_LayerCommon(cly)
        if(caffe_pb2.PoolingParameter.PoolMethod.MAX == cly.pooling_param.pool):
            layer['op'] = 'MaxPool'
        elif(caffe_pb2.PoolingParameter.PoolMethod.AVE == cly.pooling_param.pool):
            layer['op'] = 'AveragePool'
        else:
            raise NotImplementedError()
        layer['kernel_shape'] = self.get_field_hw(cly.pooling_param, 'kernel_size', 'kernel_h', 'kernel_w', [None,None])
        layer['pads'] = self.get_field_hw(cly.pooling_param, 'pad', 'pad_h', 'pad_w', [0,0])
        layer['strides'] = self.get_field_hw(cly.pooling_param, 'stride', 'stride_h', 'stride_w', [None,None])
        return layer

    def to_LayerBN(self, cly):
        layer = self.to_LayerCommon(cly)
        name = layer['name']
        params = self.caffe_model.params[name]
        C = layer['shape'][1]
        layer['scale'] = params[0].data.reshape(C)
        layer['bias'] = params[1].data.reshape(C)
        layer['var'] = np.ones((C),dtype=np.float32)
        layer['mean'] = np.zeros((C),dtype=np.float32)
        if(cly.bn_param.bn_mode == caffe_pb2.BNParameter.BNMode.INFERENCE):
            layer['bn_mode'] = 'INFERENCE'
        elif(cly.bn_param.bn_mode == caffe_pb2.BNParameter.BNMode.LEARN):
            layer['bn_mode'] = 'LEARN'
        else:
            raise NotImplementedError()
        return layer

    def to_LayerNormalize(self, cly):
        layer = self.to_LayerCommon(cly)
        name = layer['name']
        params = self.caffe_model.params[name]
        C = layer['shape'][1]
        scale = params[0].data.reshape(C)
        layer['scale'] = scale.astype(np.float32)
        return layer

    def to_LayerEltwise(self, cly):
        layer = self.to_LayerCommon(cly)
        op = self.get_field(cly.eltwise_param, 'operation', caffe_pb2.EltwiseParameter.EltwiseOp.SUM)
        if(op == caffe_pb2.EltwiseParameter.EltwiseOp.SUM):
            op = 'Add'
        else:
            raise NotImplementedError()
        layer['op']  = op
        return layer

    def to_LayerPReLU(self, cly):
        layer = self.to_LayerCommon(cly)
        params = self.caffe_model.params[layer['name']]
        layer['weights'] = params[0].data
        return layer

    def save(self, path):
        pass

    def run(self, feed, **kwargs):
        outputs = {}
        if(feed == None):
            feed = {}
            for iname in self.caffe_model.inputs:
                iname = str(iname)
                shape = self.caffe_model.blobs[iname].data.shape
                data = np.random.uniform(low=-1,high=1,size=shape).astype(np.float32)
                feed[iname] = data
        nbr = 0
        for n, v in feed.items():
            nbr = v.shape[0]
            break
        for i in range(nbr):
            for n, v in feed.items():
                self.caffe_model.blobs[n].data[...] = v[i]
            _ = self.caffe_model.forward()
            for n, v in self.caffe_model.blobs.items():
                if(n in outputs):
                    try:
                        outputs[n] = np.concatenate((outputs[n], v.data))
                    except Exception as e:
                        shape = v.data.shape
                        if((shape[-1] == 7) and 
                           (shape[0]==1) and
                           (shape[1]==1)): # this is generally DetectionOutput
                            outputs[n] = outputs[n].reshape([1,1,-1,7])
                            outputs[n] = np.concatenate((outputs[n], v.data), axis=2)
                        else:
                            raise(e)
                else:
                    outputs[n] = v.data
        for oname in self.caffe_model.outputs:
            outputs['%s_O'%(oname)] = outputs[oname]
#         for n,v in outputs.items():
#             if(len(v.shape) == 4):
#                 v = v.transpose(0,2,3,1)
#             v.tofile('tmp/%s.raw'%(n))
#         exit()
        return outputs

    def get_layers(self, names, lwnn_model):
        layers = []
        for ly in lwnn_model:
            if(ly['name'] in names):
                layers.append(ly)
        return layers

    def get_inputs(self, layer, lwnn_model):
        inputs = []
        L = list(lwnn_model)
        L.reverse()
        for iname in layer['inputs']:
            exist = False
            for ly in L:
                for o in ly['outputs']:
                    if(o == iname):
                        inputs.append(ly['name'])
                        exist = True
                        break
                if(exist):
                    break
            if(exist == False):
                raise Exception("can't find %s for %s"%(iname, layer))
        return inputs

    def convert(self):
        lwnn_model = []
        for ly in self.layers:
            op = str(ly.type)
            if(op in self.TRANSLATOR):
                translator = self.TRANSLATOR[op]
            else:
                translator = self.to_LayerCommon
            layer = translator(ly)
            lwnn_model.append(layer)
        for iname in self.caffe_model.inputs:
            iname = str(iname)
            layers = self.get_layers([iname], lwnn_model)
            if(len(layers) == 0):
                layer = LWNNLayer(name=iname, 
                          op='Input',
                          outputs=[iname],
                          shape=self.caffe_model.blobs[iname].data.shape)
                lwnn_model.insert(0, layer)
        for oname in self.caffe_model.outputs:
            oname = str(oname)
            inp = None
            for ly in lwnn_model:
                if(oname in ly['outputs']):
                    inp = ly
                    break
            layer = LWNNLayer(name=oname+'_O', 
                      op='Output',
                      inputs=[inp['name']],
                      outputs=[oname+'_O'],
                      shape=inp['shape'])
            lwnn_model.append(layer)
        for id,ly in enumerate(lwnn_model):
            if('inputs' in ly):
                inputs = self.get_inputs(ly, lwnn_model[:id])
                ly['inputs'] = inputs
        self.lwnn_model = lwnn_model

    @property
    def inputs(self):
        L = {}
        for iname in self.caffe_model.inputs:
            L[iname] = self.caffe_model.blobs[iname].data.shape
        return L

def caffe2lwnn(model, name, **kargs):
    if('weights' in kargs):
        weights = kargs['weights']
    else:
        weights = None

    converter = CaffeConverter(model, weights)

    if('feeds' in kargs):
        feeds = kargs['feeds']
    else:
        feeds = None

    if(type(feeds) == str):
        feeds = load_feeds(feeds, converter.inputs)

    model = LWNNModel(converter, name, feeds=feeds)

    model.generate()

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert caffe to lwnn')
    parser.add_argument('-i', '--input', help='input caffe model', type=str, required=True)
    parser.add_argument('-w', '--weights', help='input caffe weights', type=str, default=None, required=False)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('-r', '--raw', help='input raw directory', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-9]
    caffe2lwnn(args.input, args.output, weights=args.weights, feeds=args.raw)
