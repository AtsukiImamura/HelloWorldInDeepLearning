import sys, os
sys.path.append(os.pardir)
import numpy as np

# レイヤのインターフェースを定める基底クラス
class Layer:

    LEARNING_RATE = 0.1

    # コンストラクタ
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    # 順方向
    def forward(self):
        pass

    # 逆方向
    def backward(self):
        pass
    
    # パラメータのアップデート（誤差逆伝播法による）
    def update(self):
        pass
    
