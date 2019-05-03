import sys, os
sys.path.append(os.pardir)
import numpy as np

import Layer as Layer

from collections import OrderedDict
from AffinLayer import AffinLayer
from SoftmaxWithLoss import SoftmaxWithLoss
from Relu import Relu


class Net:

    def __init__(self):
        self.layers = OrderedDict()
        self.layers['A1'] = AffinLayer(100, 50, 1)
        self.layers['R1'] = Relu()
        self.layers['A2'] = AffinLayer(50, 20, 1)
        self.lastLayer = SoftmaxWithLoss()

    # 逆方向
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forword()
        
        return x

    # 順方向
    def forward(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 認識精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        return np.sum(y == t) / float(x.shape[0])

    # 各レイヤのパラメータをアップデートする
    def update(self):
        for layer in self.layers.values():
            layer.update()
        # self.lastLayer.update()

    

