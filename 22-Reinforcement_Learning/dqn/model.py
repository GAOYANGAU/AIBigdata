#!/usr/bin/env python3

"""
模型文件: 构建训练网络的函数放在该模块中
Contributor：huangsihan
Reviewer：xionglongfei
"""


import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions, agent_history_length, resized_width, resized_height, name_scope):
	"""
	文中第 22.6.1 章: 模型结构
	"""
	with tf.device("/cpu:0"):
		with tf.name_scope(name_scope):
			state = tf.placeholder(tf.float32, [None, agent_history_length, resized_width, resized_height], name="state")
			inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
			# 图 22-8, 第一层, 卷积层
			model = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', padding='same', data_format='channels_first')(inputs)
			# 图 22-8, 第二层, 卷积层
			model = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same', data_format='channels_first')(model)

			model = Flatten()(model)
			# 图 22-8, 第三层, 全连接层
			model = Dense(256, activation='relu')(model)
			# 图 22-8, 第四层, 全连接层
			q_values = Dense(num_actions)(model)

	m = Model(inputs=inputs, outputs=q_values)

	return state, m
