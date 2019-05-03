import sys, os
sys.path.append(os.pardir)
import numpy as np

import Layer as Layer

class Relu (Layer.Layer):
    
     def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

