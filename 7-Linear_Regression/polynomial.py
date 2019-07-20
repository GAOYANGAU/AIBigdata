"""
房屋价格线性回归
Contributor：heze
Reviewer：xionglongfei
"""

'''
某城市的房屋面积与成交的平均价格如下表所示
样本  面积（平方米）  价格（万元）
 1       50          100
 2       100         150
 3       150         200
 4       200         230
 5       250         260
 6       300         300
以下是分析过程
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# 原始数据
A = [50, 100, 150, 200, 250, 300]
P = [100, 150, 200, 230, 260, 300]

x = np.array(A).reshape(-1, 1)
y = np.array(P)

# 实例化一个二次多项式特征实例
poly_reg = PolynomialFeatures(degree=2)

# 用二次多项式对样本x值做变换
X_poly = poly_reg.fit_transform(x)

# 创建一个线性回归实例
regr = linear_model.LinearRegression()

# 以多项式变换后的x值为输入，代入线性回归模型做训练
regr.fit(X_poly, y)

# 设计x轴一系列点作为画图的x点集
data_x = np.linspace(30, 400, 100)

# 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
xx_quadratic = poly_reg.transform(data_x.reshape(data_x.shape[0], 1))
data_y = regr.predict(xx_quadratic)

# 绘图
plt.plot(x, y, 'ko', label="Original Noised Data")
plt.plot(data_x, data_y, 'r', label='Fitted Curve')
plt.show()
