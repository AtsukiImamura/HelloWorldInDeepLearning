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
        
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size

        # batch_size = self.t.shape[0]
        # if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
        #     dx = (self.y - self.t) / batch_size
        # else:
        #     dx = self.y.copy()
        #     dx[np.arange(batch_size), self.t] -= 1
        #     dx = dx / batch_size
        
        # return dx

    
    def update(self):
        pass

    def softmax(self, x):
        # c = np.max(x)
        # exp_x = np.exp(x-c)

        # return exp_x / np.sum(exp_x)

        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self, y, t):
        # if y.ndim == 1:
        #     t = t.reshape(1, t.size)
        #     y = y.reshape(1, y.size)
        
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
