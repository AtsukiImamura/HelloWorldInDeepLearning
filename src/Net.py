import sys, os
sys.path.append(os.pardir)
import numpy as np

import Layer as Layer

from Relu import Relu
from collections import OrderedDict
from AffinLayer import AffinLayer
from SoftmaxWithLoss import SoftmaxWithLoss

class Net:

    def __init__(self, input_size = 784, output_size = 10):

        self.layers = OrderedDict()
        self.layers['A1'] = AffinLayer(input_size, 50, 1)
        self.layers['R1'] = Relu()
        self.layers['A2'] = AffinLayer(50, output_size, 1)
        self.lastLayer = SoftmaxWithLoss()

    # 学習
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    # 順方向
    def loss(self, x, t):
        y = self.predict(x)
        # print('[Net] forward  sum(y)='+str(np.sum(y)))
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # print('[gradient] sum(x):'+str(np.sum(x)))
        self.loss(x, t)
        self.backward()
        self.update()

    # 認識精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        return np.sum(y == t) / float(x.shape[0])

    # def gradient(self, x, t):
    #     self.forward(x, t)
    

    def backward(self, dout = 1):
        dout = self.lastLayer.backward(dout)
        
        # print('[backward] sum(dout):'+str(np.sum(dout)))

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

    # 各レイヤのパラメータをアップデートする
    def update(self):
        for layer in self.layers.values():
            layer.update()
        # self.lastLayer.update()

    

