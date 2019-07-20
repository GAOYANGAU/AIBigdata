"""
小车速度线性回归
Contributor：heze
Reviewer：xionglongfei
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# 原始数据
N = [1, 2, 3, 4, 5, 6, 7, 8, 9]
V = [0.199, 0.389, 0.580, 0.783, 0.980, 1.177, 1.38, 1.575, 1.771]

# 转换成numpy array
x = np.array(N)
y = np.array(V)

xdata = x.reshape(-1, 1)
ydata = y.reshape(-1, 1)

# 建立线性回归模型
regr = linear_model.LinearRegression()

# 拟合
regr.fit(xdata, ydata)

# 不难得到直线的斜率、截距
a, b = regr.coef_[0][0], regr.intercept_[0]

# 画图
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, a*x+b, 'r', label='Fitted line')
plt.show()
