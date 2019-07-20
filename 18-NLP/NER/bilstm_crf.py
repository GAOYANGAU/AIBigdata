#!/usr/bin/env python3
# -*- coding=utf-8 -*-


from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF


class BiLSTM_CRF_NER(object):
    """
    Implementation of Bi-LSTM-CRF NER model by Keras.
    """
    def __init__(self, vocab_size=50000, 
                 emb_dims=100,
                 birnn_unit=200,
                 max_sent_len=200,
                 tag_num=7):
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.birnn_unit = birnn_unit
        self.max_sent_len = max_sent_len
        self.tag_num = tag_num

    def build_model(self):
        input_layer = Input(shape=(self.max_sent_len,), name='input_layer')
        embed_layer = Embedding(self.vocab_size, 
                                self.emb_dims, 
                                mask_zero=True, 
                                name='embedding_layer')(input_layer)
        birnn_layer = Bidirectional(LSTM(self.birnn_unit // 2, 
                                    return_sequences=True,
                                    name = 'bilstm_layer'))(embed_layer)
        crf = CRF(self.tag_num, sparse_target=False, name='crf_layer')
        crf_layer = crf(birnn_layer)
        model = Model(inputs=input_layer, outputs=crf_layer)
        model.compile(optimizer='adam',
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        return model
