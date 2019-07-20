"""
循环卷积网络
Contributor：zhenghui
Reviewer：xionglongfei
网络结构：单层LSTM运算单元
输入：1x784
输出：1x10
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 输入是28*28长度的向量，reshape成 28 * 28，
# 那么输入长度则是 28，时序长度也是 28
# 如果输入的是句子，每个句子 5 个单词，每个单词有64维
# 那么 input = 5, timestep = 64
input_size = 28
timestep_size = 28
# 128是个经验值
hidden_size = 128

# 标签长度，我们有10个数字
label_size = 10

# 样本batch
batch_size = 50


def main():
    mnist = input_data.read_data_sets('../dataset', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, input_size * timestep_size])
    y_ = tf.placeholder(tf.float32, shape=[None, label_size])

    y = LSTM(x)

    # 下面的评估代码和CNN的差不多
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print('step %d, accuracy %g'%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        test_batch = mnist.test.next_batch(batch_size)
        print('test accuracy %g'%accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1]}))


def LSTM(dataset):
    # 把输入转换成 None * 28 * 28 的shape，RNN按 time step 接收输入数据
    x = tf.reshape(dataset, shape=[-1, input_size, timestep_size])

    # 获取 LSTM 单元，num_units 是隐含神经元数目，是经验值
    lstm = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, forget_bias=1.0)

    # 设置初始 state，batch 大小作为参数设置，之后会自动推理出其他参数大小
    # 注意1：如果训练数据如 X 的 dimension 不正确，会抛出 InvalidArgumentError
    # 注意2：如果测试集数据的 batch size 和训练集不一致，也会抛出异常。所以最好需要动态初始化 istate
    istate = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

    # 用 dynamic_rnn 的方式执行所有时序
    y, states = tf.nn.dynamic_rnn(cell=lstm, inputs=x, initial_state=istate)

    # 全连接层，再接上 softmax 既可以作为分类输出
    weight_out = tf.Variable(tf.random_normal(shape=[hidden_size, label_size]))
    bias_out = tf.Variable(tf.random_normal(shape=[label_size]))

    return tf.matmul(y[:, -1, :], weight_out) + bias_out

if __name__ == '__main__':
    main()