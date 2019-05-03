import sys, os
sys.path.append(os.pardir)
import numpy as np

import Layer as Layer

class AffinLayer (Layer.Layer):

    # コンストラクタ
    def __init__(self, input_size, output_size, weight_init_std):
        super(AffinLayer, self).__init__(input_size, output_size)
        self.weight = weight_init_std * np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.dw = None
        self.db = None
    
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x
        out = np.dot(x, self.weight) + self.bias
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.weight.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape)
        return dx

    def update(self):
        self.weight -= self.LEARNING_RATE * self.dw
        self.bias -= self.LEARNING_RATE * self.db
    