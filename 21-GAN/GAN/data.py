# _*_ coding: utf-8 _*_
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)

def getData(size=1000):
    xs = np.random.uniform(low=-10, high=10, size=(size,))
    bs = np.random.normal(loc=0, scale=5, size=(size,))
    ys = np.power(xs, 2) + bs
    return xs, ys

def sampleFigure(xs, ys, fileName):
    import matplotlib.pyplot as plt
    l = sorted(zip(xs, np.power(xs, 2), ys))
    x_ = [x for (x, x_2, y) in l]
    x_2 = [x_2 for (x, x_2, y) in l]
    y_ = [y for (x, x_2, y) in l]

    # 输出图片大小
    plt.figure(figsize=(18,10))

    # 刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x_, x_2, 'bo', label='x_2 = x*x')
    plt.plot(x_, y_, 'ro', label='y = x*x + b')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(r'./imgs/{}.png'.format(fileName))
    plt.show()

def main():
    # 获取训练样本，并划分训练集、测试集
    xs, ys = getData(size=1000)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1)

    # 画出训练集、测试集分布，并保存图片
    sampleFigure(xs, ys, 'all')
    sampleFigure(x_train, y_train, 'train')
    sampleFigure(x_test, y_test, 'test')

if __name__ == '__main__':
    main()