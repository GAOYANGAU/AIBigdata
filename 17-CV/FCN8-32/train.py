"""
train
Contributor：lujin
Reviewer：xionglongfei
"""
from keras.callbacks import ModelCheckpoint, TensorBoard
import LoadBatches
from Models import FCN8, FCN32
import math

#############################################################################
train_images_path = "data/train_img/"
train_segs_path = "data/train_img_ann/"
###############模型训练的参数###########################
train_batch_size = 8
n_classes = 11
epochs = 100
input_height = 320
input_width = 320
#######################################################
val_images_path = "data/val_img/"
val_segs_path = "data/val_img_ann/"
val_batch_size = 8

#######使用FCN8还是FCN32##############################
key = "fcn8"
#######################################################

method = {
    "fcn32": FCN32.fcn32,
    "fcn8": FCN8.fcn8_model}

m = method[key](n_classes, input_height=input_height, input_width=input_width)

m.compile(
    loss='categorical_crossentropy',
    optimizer="adadelta",
    metrics=['acc'])

G = LoadBatches.image_segmentation_generator(train_images_path, train_segs_path, train_batch_size,
                                             n_classes=n_classes, input_height=input_height, input_width=input_width)

G_test = LoadBatches.image_segmentation_generator(val_images_path, val_segs_path, val_batch_size,
                                                  n_classes=n_classes, input_height=input_height, input_width=input_width)

checkpoint = ModelCheckpoint(
    filepath="output/%s_model.h5" %
    key,
    monitor='acc',
    mode='auto',
    save_best_only='True')
tensorboard = TensorBoard(log_dir='output/log_%s_model' % key)

m.fit_generator(generator=G,
                steps_per_epoch=math.ceil(367. / train_batch_size),
                epochs=epochs, callbacks=[checkpoint, tensorboard],
                verbose=2,
                validation_data=G_test,
                validation_steps=8,
                shuffle=True)
