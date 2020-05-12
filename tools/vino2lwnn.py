# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

from lwnn.core import *
import os
import numpy as np
from lwnn2torch import lwnn2torch
#from openvino import inference_engine as IE

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

__all__ = ['vino2lwnn']

class VinoLayer():
    def __init__(self, layer):
        self.xml = layer

    def shape(self):
        shape = []
        for port in self.xml.find('output'):
            for s in port:
                shape.append(eval(s.text))
            break
        return shape

    def ishapes(self):
        shapes = []
        for port in self.xml.find('input'):
            shape = []
            for s in port:
                shape.append(eval(s.text))
            shapes.append(shape)
        return shapes

    def outputs(self):
        outputs = []
        for port in self.xml.find('output'):
            outputs.append(port.attrib['id'])
        return outputs

    def inputs(self):
        inputs = []
        if(None != self.xml.find('input')):
            for port in self.xml.find('input'):
                inputs.append(port.attrib['id'])
        return inputs

    def blobs(self):
        blobs = []
        pmaps = {'I32':'i', 'FP32':'f'}
        if(None != self.xml.find('blobs')):
            for blob in self.xml.find('blobs'):
                if('precision' in blob.attrib):
                    dt = pmaps[blob.attrib['precision']]
                else:
                    dt = 'f'
                blobs.append((blob.tag, eval(blob.attrib['offset']), eval(blob.attrib['size']), dt))
        return blobs

    def datas(self):
        datas = {}
        data =self.xml.find('data')
        if(None != data):
            for k,v in data.attrib.items():
                try:
                    V = eval(v)
                    if(type(V) not in [list, tuple, int, float, bool]):
                        V = v
                except:
                    V = v
                datas[k] = V
        return datas

    def __getattr__(self, n):
        return self.xml.attrib[n]

class VinoConverter():
    def __init__(self, vino_model, vino_weights):
        self.vino_model = vino_model
        tree = ET.parse(vino_model)
        self.ir = tree.getroot()
        self.vino_weights = vino_weights
        self.TRANSLATOR = {
            'Input': self.to_LayerInput,
            'ScaleShift': self.to_LayerScaleShift,
            'Convolution': self.to_LayerConv,
            'Pooling': self.to_LayerPooling,
            'Eltwise': self.to_LayerEltwise,
            'Const': self.to_LayerConst,
            'Permute': self.to_LayerPermute,
            'Reshape': self.to_LayerReshape,
            'Normalize': self.to_LayerNormalize,
            'DetectionOutput': self.to_LayerDetectionOutput,
             }
        self.opMap = {
            'ReLU': 'Relu',
            'SoftMax': 'Softmax',
            }
        self.convert()

    @property
    def model(self):
        return self.lwnn_model

    def read(self, offset, num, type='f'):
        sz = 4
        tM = { 'f':np.float32, 'i':np.int32, 'q':np.int64, 'd':np.float64 }
        if(type in ['q','d']): # long long or double
            sz = 8
        num = int(num/sz)
        #assert(self.weights.tell() == offset)
        self.weights.seek(offset, 0)
        #v = np.array(struct.unpack('<'+str(num)+type, self.weights.read(sz*num)))
        v = np.ndarray(
                shape=(num,),
                dtype=tM[type],
                buffer=self.weights.read(num*sz))
        if(type == 'f'):
            v = v.astype(np.float32)
        return v

    def to_LayerCommon(self, vly, op=None):
        name = str(vly.name)
        layer = LWNNLayer(name=name,
                  outputs=vly.outputs(),
                  inputs=vly.inputs(),
                  shape=vly.shape(),
                  id=vly.id)
        if(op == None):
            op = str(vly.type)
        if(op in self.opMap):
            op = self.opMap[op]
        layer['op'] = op
        for bn, of, sz, dt in vly.blobs():
            if(bn == 'biases'):
                bn = 'bias'
            layer[bn] = self.read(of, sz, dt)
        for dn, dv in vly.datas().items():
            layer[dn] = dv
        return layer

    def to_LayerInput(self, vly):
        layer = self.to_LayerCommon(vly)
        return layer

    def to_LayerScaleShift(self, vly):
        layer = self.to_LayerCommon(vly, 'BatchNormalization')
        layer['scale'] = layer['weights']
        shape = layer['scale'].shape
        del layer['weights']
        layer['var'] = np.ones(shape, np.float32)
        layer['mean'] = np.zeros(shape, np.float32)
        return layer

    def to_LayerConv(self, vly):
        layer = self.to_LayerCommon(vly, 'Conv')
        layer['pads'] = list(layer['pads_begin']) + list(layer['pads_end'])
        if(len(layer['inputs']) > 1):
            pass # ONNX format
        else:
            group = layer['group']
            C = vly.ishapes()[0][1]
            M = layer['shape'][1]
            kernel_shape = [M, int(C/group)] + list(layer['kernel'])
            W = layer['weights']
            W = W.reshape(kernel_shape)
            layer['weights'] = W
            if('bias' not in layer):
                layer['bias'] = np.zeros((M), np.float32)
        return layer

    def to_LayerPooling(self, vly):
        layer = self.to_LayerCommon(vly)
        op = layer['pool-method']
        if(op == 'max'):
            op = 'MaxPool'
        elif(op == 'avg'):
            op = 'AveragePool'
        else:
            raise Exception('Not Supported Pooling type %s'%(op))
        layer['op'] = op
        layer['kernel_shape'] = layer['kernel']
        layer['pads'] = list(layer['pads_begin']) + list(layer['pads_end'])
        return layer

    def to_LayerEltwise(self, vly):
        layer = self.to_LayerCommon(vly)
        op = layer['operation']
        if(op == 'sum'):
            op = 'Add'
        else:
            raise Exception('Not Supported Eltwise type %s'%(op))
        layer['op'] = op
        return layer

    def to_LayerConst(self, vly):
        layer = self.to_LayerCommon(vly)
        layer['const'] = layer['custom'].reshape(layer['shape'])
        del layer['custom']
        return layer

    def to_LayerPermute(self, vly):
        layer = self.to_LayerCommon(vly, 'Transpose')
        layer['perm'] = layer['order']
        del layer['order']
        return layer

    def to_LayerReshape(self, vly):
        layer = self.to_LayerCommon(vly)
        layer['inputs'] = layer['inputs'][:1]
        return layer

    def to_LayerNormalize(self, vly):
        layer = self.to_LayerCommon(vly)
        layer['scale'] = layer['weights']
        del layer['weights']
        return layer

    def to_LayerDetectionOutput(self, vly):
        layer = self.to_LayerCommon(vly)
        code_type = layer['code_type']
        if('CENTER_SIZE' in code_type):
            code_type = 2
        elif('CORNER_SIZE' in code_type):
            code_type = 3
        else:
            code_type = 1
        layer['code_type'] = code_type
        return layer

    def save(self, path):
        pass

    def run(self, feeds, **kwargs):
        outputs = lwnn2torch(kwargs['model'], feeds)
        for n,v in outputs.items():
            if((type(v) == list) and (len(v) == 1)):
                outputs[n] = v[0]
        return outputs

    def get_layer(self, id):
        for node in self.ir:
            if(node.tag == 'layers'):
                for layer in node:
                    layer = VinoLayer(layer)
                    if(layer.id == id):
                        return layer.name
        raise Exception('layer with ID=%s not found'%(id))

    def get_inputs(self, layer, edges):
        inputs = []
        tl = layer['id']
        for tp in layer['inputs']:
            fl, _ = edges[tl][tp]
            inputs.append(self.get_layer(fl))
        return inputs

    def convert(self):
        lwnn_model = []
        edges  = {}
        self.weights = open(self.vino_weights, 'rb')
        for node in self.ir:
            if(node.tag == 'layers'):
                for layer in node:
                    layer = VinoLayer(layer)
                    op = layer.type
                    if(op in self.TRANSLATOR):
                        translator = self.TRANSLATOR[op]
                    else:
                        translator = self.to_LayerCommon
                    layer = translator(layer)
                    lwnn_model.append(layer)
            elif(node.tag == 'edges'):
                for edge in node:
                    fl, fp, tl, tp = \
                        edge.attrib['from-layer'], edge.attrib['from-port'], \
                        edge.attrib['to-layer'], edge.attrib['to-port']
                    if(tl in edges):
                        edges[tl][tp] = (fl, fp)
                    else:
                        edges[tl] = {tp:(fl, fp)}
            else:
                print('TODO: %s'%(node.tag))
        anymore = self.weights.read()
        if(len(anymore) != 0):
            raise Exception('weights %s mismatched with the model %s'%(self.vino_weights, self.vino_model))
        self.weights.close()
        for ly in lwnn_model:
            if(ly['op'] == 'Input'):
                continue
            inputs = self.get_inputs(ly, edges)
            ly['inputs'] = inputs
            if(len(ly['outputs']) == 1):
                ly['outputs'] = [ly['name']]
            else:
                ly['outputs'] = [ly['name']+o for o in ly['outputs']]
        self.lwnn_model = lwnn_model

    @property
    def inputs(self):
        L = {}
        for node in self.ir:
            if(node.tag == 'layers'):
                for layer in node:
                    layer = VinoLayer(layer)
                    if(layer.type == 'Input'):
                        L[layer.name] = layer.shape
        return L

def vino2lwnn(model, name, **kargs):
    if('weights' in kargs):
        weights = kargs['weights']
    else:
        weights = None
    converter = VinoConverter(model, weights)
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
    parser = argparse.ArgumentParser(description='convert OpenVINO IR model to lwnn')
    parser.add_argument('-i', '--input', help='input vino IR model (*.xml)', type=str, required=True)
    parser.add_argument('-w', '--weights', help='input vino IR weights (*.bin)', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('-r', '--raw', help='input raw directory', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-4]
    vino2lwnn(args.input, args.output, weights=args.weights, feeds=args.raw)
