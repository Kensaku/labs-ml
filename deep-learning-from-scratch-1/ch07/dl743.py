import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np 

#x1 = np.random.randn(1, 3, 7, 7)
x1 = np.random.randn(1, 3, 4, 4)
#col1 = im2col(x1, 5, 5, stride=1, pad=0)
col1 = im2col(x1, 3, 3, stride=1, pad=0)
print(col1.shape)

x2 = np.random.randn(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)

