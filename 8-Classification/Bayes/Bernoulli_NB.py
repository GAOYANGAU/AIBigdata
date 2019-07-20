"""
伯努利贝叶斯算法
Contributor：zhouyushun
Reviewer：xionglongfei
主要利用一组A-E功能的数据特征，预测F功能的使用情况
输入：A-E功能的数据
输出：F功能的使用情况，0,1
"""


import numpy as np
from sklearn.naive_bayes import BernoulliNB


#A-E功能使用情况
X = np.array([[1,1,1,1,0],
              [1,1,0,1,0],
              [1,0,0,1,0],
              [1,0,0,1,0],
              [1,1,0,1,0],
              [0,1,1,0,1],
              [0,0,0,1,1],
              [1,0,1,1,0],
              [1,1,0,1,0]])

#F功能的使用情况
y = np.array([0,0,0,0,0,1,1,0,0])


#把训练的数据和对应的分类放入分类器中进行训练
clf = BernoulliNB().fit(X,y)

#2018/12/19 A-E功能的使用情况
p = [[1,0,1,0,1]]

#预测 2018/12/19 F功能的使用情况
res = clf.predict(p)
print(res)


