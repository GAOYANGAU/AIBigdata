"""
高斯朴素贝叶斯算法
Contributor：zhouyushun
Reviewer：xionglongfei
主要利用鸢尾花数据集的数据特征，预测分类
输入：鸢尾花数据集的数据特征
输出：0,1,2
"""


import sklearn.datasets as skl_ds


#加载鸢尾花数据集
iris_data = skl_ds.load_iris()

#查看数据集的特征名称
print(iris_data.feature_names)
#查看分类名称
print(iris_data.target_names)

#构造训练的数据集
X = iris_data.data
y = iris_data.target

from sklearn.naive_bayes import GaussianNB

#把训练的数据和对应的分类放入分类器中进行训练
clf = GaussianNB().fit(X,y)

p = [[5.2,3.8,1.9,0.5]]

res = clf.predict(p)
print(res)



