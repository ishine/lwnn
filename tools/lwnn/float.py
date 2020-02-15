# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .base import *

class LWNNFloatC(LWNNBaseC):
    def __init__(self, model, feeds=None):
        try:
            super().__init__(model, 'float', feeds)
        except:
            LWNNBaseC.__init__(self, model, 'float', feeds)
        self.generate()

    def gen_LayerConv(self, layer):
        W = layer['weights']
        B = layer['bias']

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        isDilatedConv = False
        misc = list(layer['pads']) + strides + [self.get_activation(layer)]
        if('dilations' in layer):
            dilations = list(layer['dilations'])
            if(dilations != [1,1]):
                misc += dilations
                isDilatedConv = True
        M = np.asarray(misc, np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        if(layer['group'] == 1):
            op = 'CONV2D'
        elif(layer['group'] == layer['shape'][1]):
            op = 'DWCONV2D'
        else:
            raise Exception('convolution with group !=1 or !=C is not supported')

        if(isDilatedConv and (op == 'CONV2D')):
            op = 'DILCONV2D'
        elif(isDilatedConv and (op == 'DWCONV2D')):
            raise Exception('DWCONV2D with dilations=%s is not supported'%(dilations))

        self.fpC.write('L_{2} ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0], op))

    def gen_LayerConvTranspose(self, layer):
        W = layer['weights']
        B = layer['bias']

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        M = np.asarray(list(layer['pads']) + strides + [self.get_activation(layer)], np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        op = 'DECONV2D'
        self.fpC.write('L_{2} ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0], op))

    def gen_LayerDense(self, layer):
        W = layer['weights']
        W = W.transpose(1,0)
        B = layer['bias']

        self.gen_layer_WBM(layer, W, B)

        self.fpC.write('L_DENSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerConst(self, layer):
        self.gen_blobs(layer, [('%s_CONST'%(layer['name']), layer['const'])])
        self.fpC.write('L_CONST ({0});\n\n'.format(layer['name']))