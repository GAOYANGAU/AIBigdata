"""
世界人口线性回归
Contributor：heze
Reviewer：xionglongfei
"""

import numpy as np
from sklearn import linear_model

import matplotlib.pyplot as plt

T = [1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968]
S = [29.72, 30.61, 31.51, 32.13, 32.34, 32.85, 33.56, 34.20, 34.83]

x = np.array(T).reshape(-1, 1)
y = np.log(np.array(S))


def func(x, a, b):
    return a+b*x


# 建立线性回归模型
regr = linear_model.LinearRegression()

# 拟合
regr.fit(x, y)

# 得到拟合系数，即为直线的截距、斜率
a, b = regr.intercept_, regr.coef_[0]


# 画图
plt.plot(x, y, 'ko', label="Original Noised Data")
plt.plot(x, func(x, a, b), 'r', label='Fitted Curve')
plt.show()
