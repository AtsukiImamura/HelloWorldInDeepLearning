import sys, os
sys.path.append(os.pardir)
import numpy as np

import Layer as Layer

class SoftmaxWithLoss (Layer.Layer):

    # コンストラクタ
    def __init__(self):
        self.y = None # softmaxの出力
        self.t = None # 教師データ (one-hot vector: 正解のインデックスのものだけが１で他は０の一次元配列)

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)

        return self.loss
        

    def backword(self):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size

    def softmax(self, x):
        c = np.max(x)
        exp_x = np.exp(x-c)

        return exp_x / np.sum(exp_x)

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
