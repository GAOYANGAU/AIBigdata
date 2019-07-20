"""
肘方法做聚类个数判定
Contributor：chenronghua
Reviewer：xionglongfei
"""

import numpy as np
import sklearn.datasets as ds
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#簇距离
def calc_cluster_distance(n_clusters, cluster_centers, labels, X):
    def manhattan_distance(x, y):
        return np.sum(abs(x -y ))

    distance_sum = 0
    for i in range(n_clusters):
        #计算某个簇的点跟该簇中心的距离和
        group = labels==i
        members = X[group, :]
        for v in members:
            distance_sum += manhattan_distance(np.array(v), cluster_centers[i]) 
    return distance_sum


if __name__ == "__main__":
    #加载数据
    data_iris = load_iris()

    #归一化
    _, items = data_iris.data.shape
    for i in range(items):
        max_value = data_iris.data[:, i].max()
        data_iris.data[:, i] = data_iris.data[:, i]/max_value
    data = np.array(data_iris.data)


    #簇的数量
    n_clusters = [1, 2, 3, 4, 5, 6, 7]
    for n in n_clusters:
        cls = KMeans(n).fit(data)
        #每个簇的中心点
        cluster_centers = cls.cluster_centers_
        #每个簇所属的簇
        labels = cls.labels_     
        distances = calc_cluster_distance(n, cluster_centers, labels, data)
        print("number:{}, distance:{}".format(n, distances))                      

#输出
# number:1, distance:97.90108947482526
# number:2, distance:44.41940181784826
# number:3, distance:31.86269933456747
# number:4, distance:28.056493152930347
# number:5, distance:25.54120165869721
# number:6, distance:23.7450512533097
# number:7, distance:21.918508351970527

