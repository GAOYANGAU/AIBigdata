"""
双层卷积网络进行MNIST数字识别
Contributor：zhenghui
Reviewer：xionglongfei
网络结构：CNN x 2
输入：1x784
输出：1x10
损失函数：交叉熵
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../dataset', one_hot=True)

# 定义输入placeholder 1x784
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# W矩阵定义函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# B偏置量定义函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 首层卷积层，5x5卷积核
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# 输入数据进行reshape到28x28
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 第一个卷积层构造，经过relu激励函数得到 h_conv1 feature map
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print(h_conv1.shape) # 打印 feature map 形状
h_pool1 = max_pool_2x2(h_conv1) # relu之后过一层最大池化得到 h_pool1
print(h_pool1.shape) # 打印池化之后的输出形状

# 第二层卷积层，5x5卷积核
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 第一个卷积层构造，经过relu激励函数得到 h_conv2 feature map
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print(h_conv2.shape) # 依旧是打印形状
h_pool2 = max_pool_2x2(h_conv2) # 再第二层卷积后再过第二层的最大池化
print(h_pool2.shape) # 依旧是打印形状


# 定义一层全连接
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 对第二层池化之后的结果reshape到 7x7x64
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 与全连接进行矩阵乘法然后过relu

# 定义一层 dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 在定义最后一层全链接
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 与最后的全连接进行矩阵乘法，再过 softmax 激励函数，得到预测结果
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    session_writer = tf.summary.FileWriter('tmp/logs', sess.graph_def)
    with tf.device("/gpu:0"):
        for i in range(100):
            batch = mnist.train.next_batch(50)
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) # 每隔10个step打印一次网络准确度
                print("step %d, training accuracy %g"%(i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) # 最终取测试集进行测试