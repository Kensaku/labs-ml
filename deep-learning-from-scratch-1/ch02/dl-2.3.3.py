import numpy as np

#AND/NAND/ORは重みとバイアスだけが異なる。パーセプトロンは同じ
#XORゲートはNAND-OR-ANDの組み合わせで実現可能

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print("AND")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print("NAND")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

print("OR")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))

print("XOR")
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

"""
s1 = [0,0,1,1]
s2 = [0,1,0,1]

print("AND-AND-AND:", end='')
for i in [0, 1, 2, 3]:
    print(AND(AND(s1[i], s2[i]), AND(s1[i], s2[i])), end='')

print("")
print("AND-NAND-AND:", end='')
for i in [0, 1, 2, 3]:
    print(AND(AND(s1[i], s2[i]), NAND(s1[i], s2[i])), end='')

print("")
print("AND-OR-AND:", end='')  
for i in [0, 1, 2, 3]:
    print(AND(AND(s1[i], s2[i]), OR(s1[i], s2[i])), end='')

print("")
print("NAND-NAND-AND:", end='')
for i in [0, 1, 2, 3]:
    print(AND(NAND(s1[i], s2[i]), NAND(s1[i], s2[i])), end='')

print("")
print("NAND-OR-AND:", end='')
for i in [0, 1, 2, 3]:
    print(AND(NAND(s1[i], s2[i]), OR(s1[i], s2[i])), end='')

print("")
print("OR-OR-AND:", end='')
for i in [0, 1, 2, 3]:
    print(AND(OR(s1[i], s2[i]), OR(s1[i], s2[i])), end='')

"""
