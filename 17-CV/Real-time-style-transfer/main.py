"""
风格迁移
Contributor：lujing
Reviewer：xionglongfei
"""

import os
import tensorflow as tf

import utils as utils
from solver import Solver

'''默认参数设置'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'dir to save checkpoint in, default: ./checkpoints')

tf.flags.DEFINE_string('style_img', 'examples/style/la_muse.jpg',
                       'style image path, default: ./examples/style/la_muse.jpg')
tf.flags.DEFINE_string('train_path', './train2014',
                       'path to training images folder, default: ./train2014')
tf.flags.DEFINE_string('test_path', 'examples/content',
                       'test image path, default: ./examples/content')
tf.flags.DEFINE_string('test_dir', './examples/temp', 'test image save dir, default: ./examples/temp')

tf.flags.DEFINE_integer('epochs', 2, 'number of epochs for training data, default: 2')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for single feed forward, default: 4')

tf.flags.DEFINE_string('vgg_path', './premodel/imagenet-vgg-verydeep-19.mat',
                       'path to VGG19 network, default: ./premodel/imagenet-vgg-verydeep-19.mat')
tf.flags.DEFINE_float('content_weight', 7.5, 'content weight, default: 7.5')
tf.flags.DEFINE_float('style_weight', 100., 'style weight, default: 100.')
tf.flags.DEFINE_float('tv_weight', 200., 'total variation regularization weight, default: 200.')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate, default: 1e-3')

tf.flags.DEFINE_integer('print_freq', 100, 'print loss frequency, defalut: 100')
tf.flags.DEFINE_integer('sample_freq', 2000, 'sample frequency, default: 2000')


def check_opts(flags):
    utils.exists(flags.style_img, 'style path not found!')
    utils.exists(flags.train_path, 'train path not found!')
    utils.exists(flags.test_path, 'test image path not found!')
    utils.exists(flags.vgg_path, 'vgg network data not found!')

    assert flags.epochs > 0
    assert flags.batch_size > 0
    assert flags.print_freq > 0
    assert flags.sample_freq > 0
    assert flags.content_weight >= 0
    assert flags.style_weight >= 0
    assert flags.tv_weight >= 0
    assert flags.learning_rate >= 0

    print(flags.style_img)
    print(flags.style_img.split('/')[-1][:-4])

    style_img_name = flags.style_img.split('/')[-1][:-4]  # extract style image name
    fold_name = os.path.join(flags.checkpoint_dir, style_img_name)
    if not os.path.isdir(fold_name):
        os.makedirs(fold_name)

    fold_name = os.path.join(flags.test_dir, style_img_name)
    if not os.path.isdir(fold_name):
        os.makedirs(fold_name)


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
    check_opts(FLAGS)

    solver = Solver(FLAGS)
    solver.train()


if __name__ == '__main__':
    tf.app.run()



