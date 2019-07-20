"""
使用VOC2007对SSD300训练
Contributor：lujin
"""

import keras
import numpy as np
import pickle
from ssd300VGG16 import SSD
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from gen_data import Generator
np.set_printoptions(suppress=True)
#如果显存不够出现OOM，可以尝试控制GPU显存
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# set_session(tf.Session(config=config))


if __name__ == '__main__':
    #VOC2007中包含20类+1类背景
    NUM_CLASSES = 21
    nb_epoch = 30
    base_lr = 3e-4
    input_shape = (300, 300, 3)
    #载入默认box生成
    priors = pickle.load(open('./prior_boxes_ssd300VGG16.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    #载入gt标签
    gt = pickle.load(open('VOC2007.pkl', 'rb'))
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    path_prefix = './VOC2007/JPEGImages/'
    gen = Generator(gt, bbox_util, 16, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    model = SSD(input_shape, num_classes=NUM_CLASSES)


    #不改变freeze中层的参数
    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False
    def schedule(epoch, decay=0.9):
        return base_lr * decay ** (epoch)
    callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['acc'])

    history = model.fit_generator(gen.generate(True), gen.train_batches,
                                  nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches,
                                  workers=1)




