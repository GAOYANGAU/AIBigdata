"""
MNIST数据预处理
Contributor：yangxiangdong
Reviewer：xionglongfei
data set: MNIST,
Download url: http://yann.lecun.com/exdb/mnist/
Description of download file :
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
"""

import os
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data


class DataProcessing(object):
    def __init__(self, path):
        self.data_path = path

    def load_data(self):
        load_data = input_data.read_data_sets(self.data_path, one_hot=True)
        return load_data

    def data_description(self, data):
        mnist_data = data
        print('训练图像数据大小:{}, 训练标签数据大小:{}'.format(mnist_data.train.images.shape,
                                                mnist_data.train.labels.shape))
        print('验证图像数据大小:{},验证标签数据大小:{}'.format(mnist_data.validation.images.shape,
                                               mnist_data.validation.labels.shape))
        print('测试图像数据大小:{},测试标签数据大小:{}'.format(mnist_data.test.images.shape,
                                               mnist_data.test.labels.shape))
        print('第0张图片的向量大小：{}'.format(mnist_data.train.images[0, :].shape))
        print('第0张图片的向量表示为：\n{}'.format(mnist_data.train.images[0, :]))

    def restore_train_pictures(self, destination, mnist_data, save_image_num=10):
        save_path = destination
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(save_image_num):
            image_array = mnist_data.train.images[i, :]
            image_array = image_array.reshape(28, 28)
            save_image_name = save_path + 'img_train_' + str(i) + '.jpg'
            misc.toimage(image_array, cmin=0.1, cmax=1.0).save(save_image_name)


if __name__ == '__main__':
    data_path = './MNIST_data'
    """类初始化"""
    data_process = DataProcessing(data_path)
    """载入数据"""
    mnist_data = data_process.load_data()
    """保存训练数据原始图片"""
    save_image_path = './MNIST_data_images/'
    data_process.restore_train_pictures(save_image_path, mnist_data)
    """查看数据的大小和图片的向量表示"""
    data_process.data_description(mnist_data)
