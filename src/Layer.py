import sys, os
sys.path.append(os.pardir)
import numpy as np

class Layer:

    # コンストラクタ
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    # 順方向
    def forward(self):
        pass

    # 逆方向
    def backword(self):
        pass
    
