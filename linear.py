# 线性回归
# 梯度下降法实现

from random import random
import numpy as np
import matplotlib.pyplot as plt


# 定义损失函数 
# -\frac{1}{2m}m\sum_{i=0}^{m}{(h_{w}(x)-y^{i})^2}
# 假设函数 hw(x) = w0 + w1*x1 + w2*x2
# w 为权重
def costfunc(w0, w1, x, y):
    cost = 0
    num = len(x)
    for i in range(num):
        cost += (w0+w1*x[i]-y[i])*(w0+w1*x[i]-y[i])
    cost /= 2*num
    return cost


# 定义梯度下降函数
def grandesc(w0, w1, x, y, alpha=0.01):
    num = len(x)
    pd0, pd1 = 0, 0      # 偏导数
    for i in range(num):
        pd0 = pd0 + w0 + w1 * x[i] - y[i]
        pd1 = pd1 + (w0 + w1 * x[i] - y[i]) * x[i]
    pd0 /= num
    pd1 /= num
    # 权重更新
    w0 -= alpha * pd0
    w1 -= alpha * pd1
    return t0, t1


sample = np.zeros([20, 2])
for i in range(len(sample)):
    sample[i, 0] = random() * 20
    sample[i, 1] = sample[i, 0] * 9 - 8 + (random() - 0.5) * 100

X = sample[:, 0]
Y = sample[:, 1]




w0, w1 = 1, 1
cost = 100
times = 10000
for i in range(times):
    if cost == 0:
        break
    else:
        w0, w1 = grandesc(w0, w1, X, Y)
        cost = costfunc(w0, w1, X, Y)
        print(cost)

print(w0, w1)
F = X*w1 + w0
plt.scatter(X, Y, c='r', marker='x')
plt.plot(X, F, c='b')
plt.show()
