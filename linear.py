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


# 定义梯地下降函数
def grandesc(t0, t1, x, y, alpha=0.01):
    num = len(x)
    pd0, pd1 = 0, 0
    for i in range(num):
        pd0 = pd0 + t0 + t1 * x[i] - y[i]
        pd1 = pd1 + (t0 + t1 * x[i] - y[i]) * x[i]
    pd0 /= num
    pd1 /= num
    t0 -= alpha * pd0
    t1 -= alpha * pd1
    return t0, t1



sample = np.zeros([20, 2])
for i in range(len(sample)):
    sample[i, 0] = random() * 20
    sample[i, 1] = sample[i, 0] * 9 - 8 + (random() - 0.5) * 100

X = sample[:, 0]
Y = sample[:, 1]




t0, t1 = 1, 1
cost = np.int64(100)
times = 100000
for i in range(times):
    if cost == 0:
        break
    else:
        t0, t1 = grandesc(t0, t1, X, Y)
        cost = costfunc(t0, t1, X, Y)
        print(cost)

print(t0, t1)
F = X*t1 + t0
plt.scatter(X, Y, c='r', marker='x')
plt.plot(X, F, c='b')
plt.show()
