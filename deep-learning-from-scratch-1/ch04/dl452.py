import sys, os
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist
from dl451 import TwoLayerNet

import time

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=True)

print("end load MNIST")

train_loss_list = []

#ハイパーパラメータ
iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    start = time.time()
    print("start roop:" + str(i))

    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)

    #パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    elapsed_time = time.time() - start
    print("end roop:" + str(i))
    print("elapsed_time:{0}".format(elapsed_time))


print(train_loss_list)