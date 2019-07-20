"""
轮廓系数对kmeans进行度量
Contributor：chenronghua
Reviewer：xionglongfei
"""

import numpy as np
import sklearn.datasets as ds
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
    
#加载数据
data_iris = load_iris()

#归一化
_, items = data_iris.data.shape
for i in range(items):
    max_value = data_iris.data[:, i].max()
    data_iris.data[:, i] = data_iris.data[:, i]/max_value
X = np.array(data_iris.data)

#簇的数量
n_clusters = 3
cls = KMeans(n_clusters).fit(X)

#每个簇的中心点
cls.cluster_centers_

#X中每个簇所属的簇
cls.labels_

#曼哈顿距离
def manhattan_distance(x, y):
    return np.sum(abs(x-y))

#a[0], X[0] 到其他点的距离的平均值
distance_sum = 0
for v in X[1:]:
    distance_sum += manhattan_distance(np.array(X[0]), np.array(v))
av = distance_sum / len(X[1:])
print(av)
#0.985673814378

#b(v), X[0]
distance_min = 1000000
for i in range(n_clusters):
    group = cls.labels_ == i
    members = X[group, :]
    for v in members:
        if np.array_equal(v, X[0]):
            continue
        distance = manhattan_distance(np.array(v), cls.cluster_centers_)
        if distance_min > distance:
            distance_min = distance

bv = distance_sum / n_clusters
print(bv)
#48.9551327808

sv = float(bv -av) / max(av, bv)
print(sv)
#0.979865771812



