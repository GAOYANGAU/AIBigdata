"""
mnist 全连接代码
Contributor：zhenghui
Reviewer：xionglongfei
网络结构: 单层全连接 + Softmax
输入：1 x 784
输出: 1 x 10
损失函数：cross_entropy
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../dataset', one_hot=True)

# 定义输入 1x784 placeholder
x = tf.placeholder('float', shape=[None, 784])

# 定义全连接层
w1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]))

# 定义网络预测输出结果，1 x 10
y1 = tf.nn.softmax(tf.matmul(x, w1) + b1)

# 标签placeholder
y_label = tf.placeholder('float', shape=[None, 10])

# log记录，输出各个标签和网络输出结果的shape
print('shape y1: {}'.format(str(y1.shape)))
print('shape log(y1): {}'.format(str(tf.log(y1).shape)))
print('shape y_label: {}'.format(str(y_label.shape)))

# 交叉熵损失函数
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y_label))

# 定义梯度下降算法，step 为 0.002
train_step = tf.train.GradientDescentOptimizer(0.002).minimize(cross_entropy)

# 初始化所有变量
init = tf.initialize_all_variables()

# 定义运算会话
sess = tf.Session()
sess.run(init)

# 执行2000次迭代，每次取50个数据作为一个 batch
for i in range(2000):
    if i % 100 == 0:
        print('saving step : {}'.format(i))
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_label: batch_ys})

# 在2000次运算后定义预测值和标签之间的误差值计算方式
correct_pred = tf.equal(tf.argmax(y1, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

# 取测试数据进行模型评估
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_label:mnist.test.labels}))
