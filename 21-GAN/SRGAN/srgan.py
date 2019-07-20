# _*_ coding: utf-8 _*_

"""
Super-Resolution Using a Generative Adversarial Network
Contributor：haikuoxin
Reviewer：xionglongfei
"""


import os
import sys
import datetime
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class SRGAN():
    def __init__(self):
        # 定义shape
        self.channels = 3
        self.lr_height = 56
        self.lr_width = 56
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4
        self.hr_width = self.lr_width*4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # 生成器中残差模块个数
        self.n_residual_blocks = 16

        optimizer = Adam(0.0002, 0.5)

        # 用预训练的 vgg 为 high_sr 和 fake high_sr 提取特征， 并训练
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 数据提取器
        self.data_loader = DataLoader(img_res=(self.hr_height, self.hr_width))

        # 按照PatchGAN的算法计算判别器的输出
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # G 和 D 第一层的 filters 数目
        self.gf = 64
        self.df = 64

        # 创建并编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 创建生成器
        self.generator = self.build_generator()

        # 定义高低分辨率的输入
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # 通过生成器以低分辨率图片生成高分辨率图片
        fake_hr = self.generator(img_lr)

        # 使用预训练的vgg从生成的高分辨率图片中提取特征
        fake_features = self.vgg(fake_hr)

        # 对于联合模型， 仅更新生成器的参数
        self.discriminator.trainable = False

        # 判别器判别生成器生成图片的真假
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)


    def build_vgg(self):
        """
        定义一个预训练的vgg模型，用于从高分辨率图片中提取特征
        """
        vgg = VGG19(weights="imagenet")
        # 将输出设置为vgg19第三个残差模块的最后一个卷积层的输出
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # 提取图片特征
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # 低分辨率图片的输入
        img_lr = Input(shape=self.lr_shape)

        # 前残差块
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # 16个残差模块
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # 后残差快
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # 上采样
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # 生成高分辨率图像
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # 判别器输入
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        d_losses, g_losses = [], []
        for epoch in range(epochs):

            # ----------------------
            #  训练判别器
            # ----------------------

            # 抽样batch_size大小的数据
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # 由低分辨率图像生成高分辨率图像
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # 更新判别器参数，计算判别器loss
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  训练生成器
            # ------------------

            # 抽样batch_size大小的数据
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # 生成器尽可能让判别器对生成图片的判别结果为True
            valid = np.ones((batch_size,) + self.disc_patch)

            # 利用预训练的vgg19从ground truth 的高清图片中提取特征
            image_features = self.vgg.predict(imgs_hr)

            # 训练生成器
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # 打印和收集loss
            print ("%d==> d_loss: %s | g_loss: %s | time: %s" % (epoch, d_loss, g_loss, elapsed_time))
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            # 每过一定轮次， 保存一次生成的图片。
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        
        # 保存loss
        loss_df = pd.DataFrame({'d_loss':d_losses,
                                'g_loss':g_losses})
        loss_df.to_csv(r'./outputs/loss_df.csv', index=False)
        print ('save done!')

    def sample_images(self, epoch):
        os.makedirs('./images', exist_ok=True)
        r = 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # 调节图片像素值 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # 保存原始的高清图片
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_hr[i])
            fig.savefig('images/%d_%d_imgs_hr.png' % (epoch, i))
            plt.close()

        # 保存生成的高清图片
        for i in range(r):
            fig = plt.figure()
            plt.imshow(fake_hr[i])
            fig.savefig('images/%d_%d_fake_hr.png' % (epoch, i))
            plt.close()

        # 保存低清图片
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%d_%d_lowres.png' % (epoch, i))
            plt.close()

if __name__ == '__main__':
    gan = SRGAN()
    print (gan.generator.summary())
    print (gan.discriminator.summary())
    gan.train(epochs=1000*100, batch_size=8, sample_interval=100)
