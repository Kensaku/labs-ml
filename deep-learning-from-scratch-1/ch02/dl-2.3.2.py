import numpy as np 

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7

reg1 = x * w
reg2 = np.sum(x * w)
reg3 = b + np.sum(x * w)

print(reg1, reg2, reg3)


