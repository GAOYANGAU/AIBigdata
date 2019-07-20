"""
Density-Based Spatial Clustering of Applications with Noise
该代码中使用Kmeans和agg与DBSCAN做对比，仅有DBSCAN能够聚类成功
Contributor：chenronghua
Reviewer：xionglongfei
"""

import sklearn.datasets as ds
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pylab as plt

#DBSCAN 可以发现各种形状的簇，列如下面这组数据

def show_data(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()


#生成互相重叠的半月形数据集
N = 1000
X, Y = ds.make_moons(n_samples=N, noise=.05)


#显示原始数据
show_data(X, Y)

#KMeans聚类
kmeans_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
print("kmeans acc:%s"% adjusted_rand_score(Y, kmeans_pred))
show_data(X, kmeans_pred)

#层次聚类
agg_cls = AgglomerativeClustering(n_clusters=2, linkage='average')
agg_pred = agg_cls.fit_predict(X)
print("agg acc:%s"% adjusted_rand_score(Y, agg_pred))
show_data(X, agg_pred)

#密度聚类
model = DBSCAN(eps=0.2, min_samples=4)
dbscan_pred = model.fit(X).labels_
print("dbscan acc:%s"% adjusted_rand_score(Y, dbscan_pred))
show_data(X, dbscan_pred)

