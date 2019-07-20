"""
KMeans聚类
Contributor：chenronghua
Reviewer：xionglongfei
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

#横排显示3维数据集， n表示一行有多少列，i表示当前是第几列
def show_data(X, Y, n, i, title):
    ax = plt.subplot(1, n, i, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    ax.set_xlabel('sepal length')
    ax.set_ylabel('petal width')
    ax.set_zlabel('petal width)
    ax.set_title(title)

# load data
data = load_iris()

# 输出鸢尾花的特征名称，以及种类名称
# print('iris data feature_names:{}, labels:{}'.format(data.feature_names, data.target_names))

#因为4维数据不好显示，所以使用后数据集的后[1,2,3] 3列作为测试集数据

#使用Kmeans进行预测
n_clusters = 3
X = data.data[:, [1, 2, 3]]
Y = data.target
cls = KMeans(n_clusters).fit(X)

#计算准确率
print("acc:%s"% adjusted_rand_score(Y, cls.labels_))

#显示原始数据与预测数据的对比
plt.figure(figsize=(18,8))
#原始记录
show_data(X, cls.labels_, 2, 1, "raw data")
#预测后的记录
show_data(data.data[:, [1, 2, 3]], data.target, 2, 2, "predicted data")
plt.show()
