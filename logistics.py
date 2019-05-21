# 逻辑斯谛回归

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 交叉熵损失函数
def costfunc(tran_x, tran_y, w):
    sample_num = len(tran_y)
    hw = sigmoid(np.matmul(tran_x, w))
    return -1 * (np.matmul(tran_y.T, np.log(hw)) + np.matmul((1 - tran_y.T), np.log(1 - hw))) / sample_num

# 梯度下降函数
def granddesc(tran_x, tran_y, w, alpha=0.0005):
    sample_num = len(tran_y)
    hw = sigmoid(np.matmul(tran_x, w))
    pd = np.matmul(tran_x.T, hw - tran_y) / sample_num
    nw = w - alpha * pd
    return nw


iris = datasets.load_iris()  # 引入 iris 数据集
x1 = iris.data[0:100, 2].reshape([100, 1])
x2 = iris.data[0:100, 3].reshape([100, 1])
tran_y = iris.target[0:100].reshape([100, 1])
ones = np.ones([100, 1])
tran_X = np.hstack((ones, x1, x2))


times = 10000
alpha = 10
w = np.array([[-1], [1], [-1]])
for i in range(times):
    print(costfunc(tran_X, tran_y, w))
    w = granddesc(tran_X, tran_y, w, alpha)

tx1 = np.linspace(0, 5)
ty = -(tx1 * w[1] + w[0]) / w[2]
plt.scatter(x1[0:50], x2[0:50], marker='o')
plt.scatter(x1[50:100], x2[50:100], marker='x')
plt.plot(tx1, ty)
plt.show()



