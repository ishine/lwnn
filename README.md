# LWNN - Lightweight Neural Network

Mostly, inspired by [NNOM](https://github.com/majianjia/nnom), [CMSIS-NN](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN), I want to do something for Edge AI.

But as I think NNOM is not well designed for different runtime, CPU/DSP/GPU/NPU etc, it doesn't have a clear path to handle different type of runtime, and nowdays, I really want to study somehing about OpenCL, and I came across [MACE](https://github.com/XiaoMi/mace/tree/master/mace/ops/opencl/cl), and I find there is a bunch of CL kernels can be used directly.

So I decieded to do something meaningfull, do some study of OpenCL and at the meantime to create a Lightweight Neural Network that can be suitale for decices such as PC, mobiles and MCU etc.

## Architecture

And for the purpose to support variant Deep Learning frameworks such as tensorflow/keras/caffe2, pytorch etc, the [onnx](https://onnx.ai/) will be supported by lwnn.

![arch](docs/arch.png)

| Layers/Runtime | cpu float | cpu q8 | cpu q16 | opencl | comments |
| - | - | - | - | - | - |
| Conv1D | Y | Y | Y | Y | based on Conv2D |
| Conv2D | Y | Y | Y | Y | |
| DepthwiseConv2D | Y | Y | Y | Y | |
| EltmentWise Max | Y | Y | Y | Y | |
| ReLU | Y | Y | Y | Y | |
| MaxPool | Y | Y | Y | Y | |
| Dense | Y | Y | Y | Y | |
| Softmax | Y | Y | Y | Y | |
| Reshape | Y | Y | Y | Y | |
| Pad | Y | Y | Y | Y | |
| BatchNorm | Y | Y | Y | Y | only support BatchNorm after Conv2D |

## Development

### prepare environment
```sh
conda create -n lwnn python=3.6
source activate lwnn
conda install scons tensorflow keras 
conda install -c caffe2 caffe2
pip install keras2onnx onnxruntime
```

### build

```sh
scons
```
