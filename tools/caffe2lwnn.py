# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn import *
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

__all__ = ['caffe2lwnn']

class CaffeConverter():
    def __init__(self, caffe_model, caffe_weights):
        self.net = caffe_pb2.NetParameter()
        text_format.Merge(open(caffe_model,'r').read(),self.net)
        self.model = caffe.Net(caffe_model,caffe_weights,caffe.TEST)
        if(len(self.net.layer)==0):    #some prototxts use "layer", some use "layers"
            self.layers = self.net.layers
        else:
            self.layers = self.net.layer
        self.TRANSLATOR = {
            'Convolution': self.to_LayerConv,
            'Permute': self.to_LayerTranspose,
            'PriorBox': self.to_LayerPriorBox,
            'DetectionOutput': self.to_LayerDetectionOutput,
            'Concat': self.to_LayerConcat,
             }
        self.opMap = { 
            'ReLU': 'Relu', 
            'Flatten': 'Reshape',
            }

    def to_LayerCommon(self, cly, op=None):
        name = str(cly.name)
        blob = self.model.blobs[cly.top[0]]
        layer = { 'name':name,
                  'outputs': [str(o) for o in cly.top],
                  'inputs': [str(o) for o in cly.bottom],
                  'shape': list(blob.data.shape)
                }
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

    def to_LayerConv(self, cly):
        layer = self.to_LayerCommon(cly, 'Conv')
        name = layer['name']
        params = self.model.params[name]
        layer['weights'] = params[0].data
        layer['bias'] = params[1].data
        stride = self.get_field(cly.convolution_param, 'stride', [1])
        pad = self.get_field(cly.convolution_param, 'pad', [0])
        if(len(stride)==1):
            strideW  = strideH = stride[0]
        else:
            strideW  = strideH = 1
        if(len(pad)==1):
            padW  = padH = pad[0]
        else:
            padW  = padH = 1
        layer['strides'] = [strideW,strideH]
        layer['pads'] = [padW,padH,0,0]
        layer['group'] = self.get_field(cly.convolution_param, 'group', 1)
        return layer

    def to_LayerTranspose(self, cly):
        layer = self.to_LayerCommon(cly, 'Transpose')
        layer['perm'] = [d for d in cly.permute_param.order]
        return layer

    def to_LayerPriorBox(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['min_size'] = cly.prior_box_param.min_size[0]
        layer['aspect_ratio'] = cly.prior_box_param.aspect_ratio[0]
        layer['variance'] = [v for v in cly.prior_box_param.variance]
        layer['offset'] = cly.prior_box_param.offset
        layer['flip'] = cly.prior_box_param.flip
        layer['clip'] = cly.prior_box_param.clip
        return layer

    def to_LayerDetectionOutput(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['num_classes'] = cly.detection_output_param.num_classes
        layer['nms_threshold'] = cly.detection_output_param.nms_param.nms_threshold
        layer['top_k'] = cly.detection_output_param.nms_param.top_k
        layer['code_type'] = cly.detection_output_param.code_type
        layer['keep_top_k'] = cly.detection_output_param.keep_top_k
        layer['confidence_threshold'] = cly.detection_output_param.confidence_threshold
        layer['share_location'] = cly.detection_output_param.share_location
        layer['background_label_id'] = cly.detection_output_param.background_label_id
        shape = layer['shape']
        layer['shape'] = [shape[0],
                          shape[3],
                          cly.detection_output_param.num_classes*2 if (shape[2]==1) else shape[2],
                          shape[1]]
        return layer

    def to_LayerConcat(self, cly):
        layer = self.to_LayerCommon(cly)
        layer['axis'] = cly.concat_param.axis
        return layer

    def save(self, path):
        pass

    def run(self, feed=None):
        outputs = {}
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
            for ly in L:
                for o in ly['outputs']:
                    if(o == iname):
                        inputs.append(ly['name'])
                        if(len(inputs) == len(layer['inputs'])):
                            # caffe may reuse top buffer to save memory
                            return inputs
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
        for iname in self.model.inputs:
            iname = str(iname)
            layers = self.get_layers([iname], lwnn_model)
            if(len(layers) == 0):
                layer = { 'name': iname, 
                          'op': 'Input',
                          'outputs' : [iname],
                          'shape': self.model.blobs[iname].data.shape }
                lwnn_model.insert(0, layer)
        for oname in self.model.outputs:
            oname = str(oname)
            inp = None
            for ly in lwnn_model:
                if(oname in ly['outputs']):
                    inp = ly
                    break
            layer = { 'name': oname+'_O', 
                      'op': 'Output',
                      'inputs' : [inp['name']],
                      'outputs' : [oname+'_O'],
                      'shape': inp['shape'] }
            lwnn_model.append(layer)
        for id,ly in enumerate(lwnn_model):
            if('inputs' in ly):
                inputs = self.get_inputs(ly, lwnn_model[:id])
                ly['inputs'] = inputs
        return lwnn_model

def caffe2lwnn(model, name, **kargs):
    if('weights' in kargs):
        weights = kargs['weights']
    else:
        weights = None
    model = LWNNModel(CaffeConverter(model, weights), name)
    if('feeds' in kargs):
        feeds = kargs['feeds']
    else:
        feeds = None

    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to lwnn')
    parser.add_argument('-i', '--input', help='input caffe model', type=str, required=True)
    parser.add_argument('-w', '--weights', help='input caffe weights', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-9]
    caffe2lwnn(args.input, args.output, weights=args.weights)
