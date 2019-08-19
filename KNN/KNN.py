import numpy as np
import math
from collections import Counter
from sklearn.datasets import load_iris

'''
遍历KNN算法
'''
class KNN:
    """
    K近邻算法
    """ 
    def __init__(self, train_x, train_y, k=3, p=2):
        """
        k: 临近点树
        p: 距离度量
        """
        self.k = k
        self.p = p
        self.train_x = train_x
        self.train_y = train_y
    def predict(self, X):
        knn_list=[]
        "先将前n个train_x存入knn列表"
        for i in range(self.k):
            dist = np.linalg.norm(X - train_x[i], ord = self.p)
            knn_list.append((dist, train_y[i]))
        
        "从训练样本中挑选离X最近的点"
        for i in range(len(train_x)-self.k):
            disk_list = [knn_list[i][0] for i in range(len(knn_list))]
            max_index = disk_list.index(max(disk_list))
            dist = np.linalg.norm(X - train_x[i+self.k], ord = self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, train_y[i+self.k])
        
        y = [knn_list[i][1] for i in range(len(knn_list))]
        count_pairs = Counter(y)
        max_count = sorted(count_pairs.items(), key = lambda x:x[1])[-1][0]
        return max_count
