"""
Agglomerative Clustering
Contributor：chenronghua
Reviewer：xionglongfei
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

#显示3维数据集
def show_data(X, Y, n, i, title):
    ax = plt.subplot(1, n, i, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    ax.set_xlabel('sepal length')
    ax.set_ylabel('petal width')
    ax.set_zlabel('petal width')
    ax.set_title(title)

# load data
data = load_iris()

# 输出鸢尾花的特征名称，以及种类名称
# print('iris data feature_names:{}, labels:{}'.format(data.feature_names, data.target_names))

#因为4维数据不好显示，所以使用后数据集的后[0, 1, 2] 3列作为测试集数据
X = data.data[:, [0, 1, 2]]
Y = data.target

#不同相似度算法对结果的影响, 可以看到同一个数据集不同的类间相似度聚类结果差别还是蛮大的
linkages = ['ward','complete','average']
plt.figure(figsize=(18,5))
linkage_num = len(linkages)
for i in range(linkage_num):
    linkage = linkages[i]
    cls = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    pred = cls.fit_predict(X)
    show_data(X, pred,linkage_num, i+1, 'linkage:{}, acc:{}'.format(linkage, adjusted_rand_score(Y, pred)))
plt.show()


