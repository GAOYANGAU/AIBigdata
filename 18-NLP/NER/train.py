#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train NER model.
Contributor：songchao
Reviewer：xionglongfei
"""


import json
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from bilstm_crf import BiLSTM_CRF_NER

# read word index
with open('data/word_index.json', encoding='utf8') as f:
    word_index = json.load(f)

# read tag index
with open('data/tag_index.json', encoding='utf8') as f:
    tag_index = json.load(f)

# load word_index
token = Tokenizer()
token.word_index = word_index

def load_data(fname, tag2id):
    # read raw data
    with open(fname, encoding='utf8') as f:
        split_tag = '\n\n'
        content = f.read().strip().split(split_tag)
        data = [[line.split() for line in s.split('\n')] for s in content]
    text_seq = [[x[0] for x in line] for line in data]
    tag_seq = [[x[1] for x in line] for line in data]
    # tokenize data
    text_token = token.texts_to_sequences(text_seq)
    tag_token = [[tag2id[t] for t in line] for line in tag_seq]
    # padding data
    text_pad = pad_sequences(text_token, maxlen=200)
    tag_pad = pad_sequences(tag_token, maxlen=200)
    tag_pad = K.eval(K.one_hot(tag_pad, len(tag2id)))
    return text_pad, tag_pad

# load data
x_train, y_train = load_data('data/train_data.data', tag_index)
x_test, y_test = load_data('data/test_data.data', tag_index)

# build model
bilstm_crf = BiLSTM_CRF_NER()
model = bilstm_crf.build_model()
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test))
    
    