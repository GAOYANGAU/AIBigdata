#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train TextCNN model.
Contributor：songchao
Reviewer：xionglongfei
"""


from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence
from textcnn import TextCNN

# model hyperparameter
MAX_SENT_LEN = 400
MAX_WORD_NUM = 50000
EMBEDDING_DIMS = 100
CLASS_NUM = 1
LAST_ACTIVATION = 'sigmoid'

# training hyperparameter
BATCH_SZIE = 128
EPOCHS = 10

# load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORD_NUM)

# padding sequence
x_train = sequence.pad_sequences(x_train, maxlen=MAX_SENT_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SENT_LEN)

# build model
model = TextCNN(max_sent_len=MAX_SENT_LEN, 
                max_word_num=MAX_WORD_NUM, 
                embedding_dims=EMBEDDING_DIMS,
                class_num=CLASS_NUM,
                last_activation=LAST_ACTIVATION).build_model()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# train
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=BATCH_SZIE,
          epochs=EPOCHS,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

# save model
# model.save('textcnn_model.h5')