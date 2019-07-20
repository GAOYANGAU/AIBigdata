"""
CNN版MNIST
Contributor：yangxiangdong
Reviewer：xionglongfei
data set: MNIST,
如果tensorflow自动下载失败,建议去该网站http://yann.lecun.com/exdb/mnist/
Description of download file :
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
如果读者想要了解Mnist数据集的大小,维度描述,或者保存训练数据原始图片,可以执行mnist_data_processing.py脚本,
原始图片会保存到：MNIST_data_images文件夹下
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


LEARNING_RATE = 0.001
BATCH_SIZE = 50
EPOCH = 2000


class CNN(object):
    def weight_variable(self, shape, regularization):
        """
        1.定义权重变量，从截断的正态分布中输出随机值。 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
        2.定义正则惩罚项
        :param shape: 维度
        :param regularization: 正则惩罚项
        :return:
        """
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        if regularization == 'L2':
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.05)(
                weight))  # add_to_collection()函数将新生成变量的L2正则化损失加入集合losses
            return weight
        elif regularization == 'L1':
            tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(0.01)(
                weight))  # add_to_collection()函数将新生成变量的L1正则化损失加入集合losses
            return weight
        else:
            return weight

    def bias_variable(self, shape):
        """定义偏置为常量"""
        b = tf.Variable(tf.constant(0.1, shape=shape))
        return b

    def convolution(self, x, conv_weight):
        """定义卷积"""
        return tf.nn.conv2d(x, conv_weight, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, x):
        """定义最大池化层"""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def cost_function(self, y_prd, y_real, regularization=''):
        """
        定义损失函数，由于是分类问题采用交叉熵损失函数。
        :param y_prd: 模型输出标签
        :param y_real: 实际标签
        :param regularization: 添加正则化
        :return:
        """
        cost_function = tf.reduce_mean(-tf.reduce_sum(y_real * tf.log(y_prd),
                                                      axis=1))
        if regularization in ['L1', 'L2']:
            print('apply regularizer:{}'.format(regularization))
            tf.add_to_collection('losses', cost_function)
            cost_function = tf.add_n(tf.get_collection('losses'))
        return cost_function

    def optimizer(self, lr, cost_func):
        """
        使用梯度下降最小化loss的优化方法，进行迭代。
        :param lr: 学习率
        :param cost_func: 损失函数
        :return:
        """
        train_iteration = tf.train.GradientDescentOptimizer(lr).minimize(cost_func)
        return train_iteration

    def batch_norm_wrapper(self, inputs, axes, is_training, decay=0.999):
        """
        定义batch normalization装饰器函数,在训练和预测不同阶段采用的计算方法不同，在预测阶段直接使用样本均值和方差
        :param inputs: 输入值
        :param is_training: 是否是训练阶段
        :param decay: decay反映了当前估计的衰减速度，decay越小衰减越快（指数衰减），对训练后期的batch mean有更多重视，如果数据集够大，采用接近1的值
        :return:
        """
        epsilon = 1e-3
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype=tf.float32, trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype=tf.float32, trainable=False)

        def batch_norm_training():
            batch_mean, batch_var = tf.nn.moments(inputs, axes)
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, epsilon)

        def batch_norm_inference():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
        return batch_normalized_output

    def model(self, x, is_training, keep_prob, batch_normalization, regularization):
        """
        构建两层卷积层，两层全连接层的CNN网络
        :param is_training: batch_normalization是否为训练阶段
        :param x:
        :param keep_prob: dropout保留率
        :param batch_normalization: 是否使用batch normalization
        :param regularization: 添加正则化
        :return:
        """
        """第一层卷积"""
        w_conv1 = self.weight_variable([5, 5, 1, 32], regularization)
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(x, [-1, 28, 28, 1])  # 由于使用卷积，需要把n维列向量，转为原图为28X28的矩阵表示
        h_conv1 = self.convolution(x_image, w_conv1) + b_conv1
        if batch_normalization:
            print('h_conv1 使用batch normalization')
            h_conv1 = self.batch_norm_wrapper(h_conv1, [0, 1, 2], is_training)
        h_conv1 = tf.nn.relu(h_conv1)
        h_pool1 = self.max_pool(h_conv1)
        """第二层卷积"""
        w_conv2 = self.weight_variable([5, 5, 32, 64], regularization)
        b_conv2 = self.bias_variable([64])
        h_conv2 = self.convolution(h_pool1, w_conv2) + b_conv2
        if batch_normalization:
            print('h_conv2 使用batch normalization')
            h_conv2 = self.batch_norm_wrapper(h_conv2, [0, 1, 2], is_training)
        h_conv2 = tf.nn.relu(h_conv2)
        h_pool2 = self.max_pool(h_conv2)
        """第一层全连接层"""
        w_fc1 = self.weight_variable([7 * 7 * 64, 256], regularization)
        b_fc1 = self.bias_variable([256])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1_z = tf.matmul(h_pool2_flat, w_fc1) + b_fc1
        if batch_normalization:
            """使用batch normalization"""
            print('h_fc1 使用batch normalization')
            h_fc1_z = self.batch_norm_wrapper(h_fc1_z, [0], is_training)
        h_fc1 = tf.nn.relu(h_fc1_z)
        """dropout"""
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
        """第二层全连接层"""
        w_fc2 = self.weight_variable([256, 10], regularization='')
        b_fc2 = self.bias_variable([10])
        h_fc2_z = tf.matmul(h_fc1, w_fc2) + b_fc2
        y_prd = tf.nn.softmax(h_fc2_z)

        return y_prd

    def train(self, mnist_data, droup_out=False, batch_normalization=False, early_stopping=False, regularization='L2'):
        """
        :param mnist_data:
        :param droup_out: 是否使用droupout，默认不使用
        :param early_stopping: 是否使用early_stopping，默认不使用
        :param batch_normalization:是否使用batch normalization，默认不使用若使用，建议调大学习率
        :param regularization: 添加正则， 默认为L2正则
        :return:
        """
        """设置占位符"""
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_input')
        is_training = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder(tf.float32)
        """构建网络"""
        prediction = self.model(x, is_training=is_training, keep_prob=keep_prob,
                                batch_normalization=batch_normalization, regularization=regularization)
        """使用L2正则化"""
        loss = self.cost_function(y_prd=prediction, y_real=y_input, regularization=regularization)

        train_step = self.optimizer(LEARNING_RATE, loss)
        """计算准确率"""
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        """设置训练追踪参数"""
        total_train_epoch = 0
        last_improvement_epoch = 0
        best_validation_accuracy = 0.0
        early_stopping_epoch = 300  # 300轮在验证集准确率未提升则提前终止训练
        """最优模型的变量保存"""
        saver = tf.train.Saver()
        save_dir = './cnn_checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model_save_path = os.path.join(save_dir, 'best_validation')

        """
        构建会话并输出训练结果,默认使用dropout，若训练不使用dropout可以设置参数drop_out为False.
        注意：在验证集和测试集上不使用dropout
        """
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(EPOCH):
                total_train_epoch += 1
                batch_x, batch_y = mnist_data.train.next_batch(BATCH_SIZE)
                if droup_out:
                    sess.run(train_step,
                             feed_dict={x: batch_x, y_input: batch_y, keep_prob: 0.5, is_training: True})
                else:
                    sess.run(train_step, feed_dict={x: batch_x, y_input: batch_y, keep_prob: 1.0, is_training: True})
                if (total_train_epoch % 50 == 0) or (i == (EPOCH - 1)):
                    if droup_out:
                        train_accuracy = sess.run(accuracy,
                                                  feed_dict={x: batch_x, y_input: batch_y, keep_prob: 1.0,
                                                             is_training: False})
                    else:
                        train_accuracy = sess.run(accuracy,
                                                  feed_dict={x: batch_x, y_input: batch_y, keep_prob: 1.0,
                                                             is_training: False})
                    validation_accuracy = sess.run(accuracy,
                                                   feed_dict={x: mnist_data.validation.images,
                                                              y_input: mnist_data.validation.labels, keep_prob: 1.0,
                                                              is_training: False})
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        last_improvement_epoch = total_train_epoch
                        saver.save(sess=sess, save_path=best_model_save_path)
                        improved_str = '***'
                    else:
                        improved_str = ''
                    print(
                        'Epoch:{}, train_accuracy:{}, val_accuracy:{}{}'.format(i + 1, train_accuracy,
                                                                                validation_accuracy,
                                                                                improved_str))
                if early_stopping:
                    if total_train_epoch - last_improvement_epoch > early_stopping_epoch:
                        print(
                            "No improvement found in {} iterations, stopping optimization.".format(
                                early_stopping_epoch))
                        break


if __name__ == '__main__':
    data_path = './MNIST_data'
    mnist_data = input_data.read_data_sets(data_path, one_hot=True)
    CNN().train(mnist_data)
