import sys, os
sys.path.append(os.pardir)

import numpy as np

def side_effect(arr):
    arr = arr*2
    print('side effect!')
    print(arr)

arr = np.array([[0, 1], [3, 4]])
print(arr)

side_effect(arr)
print(arr)