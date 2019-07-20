# _*_ coding: utf-8 _*_

"""
Generative Adversarial Nets
Contributor：haikuoxin
Reviewer：xionglongfei
"""


import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
tf.logging.set_verbosity(tf.logging.ERROR)

import data

class GAN():
    def __init__(self, LR, EPOCHS, BATCH_SIZE, SAMPLE_INYERVAL):
        self.LR = LR
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.SAMPLE_INYERVAL = SAMPLE_INYERVAL

        self.optimizer = Adam(self.LR)

        # 创建判别器并compile
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        # 创建生成器
        self.generator = self.build_generator()

        # 生成器生成 y_fake
        x = keras.Input(shape=(1,))
        y_fake = self.generator(x)

        # 只训练生成器
        self.discriminator.trainable = False

        # 判别器把生成器的输出 y_fake 当做输入， 并输出判别结果
        validity = self.discriminator(y_fake)

        # 联合模型
        self.combined = keras.Model(x, validity)
        self.combined.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer
        )

    def build_generator(self):

        inputs = keras.Input(shape=(1,))
        x = layers.Dense(8, activation='relu')(inputs)
        x = layers.Dense(16, activation='relu')(x)
        y_fake = layers.Dense(1, activation='linear')(x)

        generator = keras.Model(inputs, y_fake)
        return generator

    def build_discriminator(self):

        inputs = keras.Input(shape=(1,))
        x = layers.Dense(8, activation='relu')(inputs)
        x = layers.Dense(16, activation='relu')(x)
        validity = layers.Dense(1, activation='sigmoid')(x)

        discriminator = keras.Model(inputs, validity)
        return discriminator

    def train(self):
        # 获取数据，并分割数据集
        xs, ys = data.getData(size=1000)
        x_train, _, y_train, _ = train_test_split(xs, ys, test_size=0.1)

        # 画出数据分布 
        # data.sampleFigure(xs, ys, 'all')

        # 对抗样本的标签（ground truths）
        valid = np.ones((self.BATCH_SIZE, 1))
        fake = np.zeros((self.BATCH_SIZE, 1))

        d_losses, g_losses = [], []
        for epoch in range(self.EPOCHS):

            """训练判别器"""

            # 选择样本
            idx = np.random.randint(0, x_train.shape[0], self.BATCH_SIZE)
            x_bs, y_bs = x_train[idx], y_train[idx]
            
            # 生成 y_fake
            y_fake = self.generator.predict(x_bs)

            # 训练判别器
            d_loss_real = self.discriminator.train_on_batch(y_bs, valid)
            d_loss_fake = self.discriminator.train_on_batch(y_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
        
            """训练生成器"""
            g_loss = self.combined.train_on_batch(x_bs, valid)

            # 打印训练状态
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            if epoch % self.SAMPLE_INYERVAL == 0:
                print ('{:d} [d_loss: {:f}, acc: {:f}] [g_loss: {:f}]'.format(epoch, d_loss[0], d_loss[1], g_loss))
        return d_losses, g_losses

if __name__ == "__main__":
    gan = GAN(LR=1e-3, EPOCHS=1000, BATCH_SIZE=64, SAMPLE_INYERVAL=20)
    d_losses, g_losses = gan.train()