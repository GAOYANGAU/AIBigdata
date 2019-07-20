#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class TextCNN(object):
    """
    Implementation of TextCNN algorithm by Keras.
    """
    def __init__(self, max_sent_len=50,
                       max_word_num=50000,
                       embedding_dims=100,
                       class_num=10,
                       last_activation='softmax',
                       word_vector_matrix=None):
        self.max_sent_len = max_sent_len
        self.max_word_num = max_word_num
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        # if you have pre-trained word vectors:
        self.word_vector_matrix = word_vector_matrix
    
    def build_model(self):
        input_layer = Input(shape=(self.max_sent_len,), name='input_layer')
        embed_layer = Embedding(self.max_word_num, 
                                self.embedding_dims, 
                                input_length=self.max_sent_len,
                                name='embedding_layer_rand')(input_layer)
        # You can try multichannel as same as origin paper if you have pre-trained word vectors.
        if self.word_vector_matrix:
            embed_layer_static = Embedding(self.max_word_num,
                                           self.embedding_dims, 
                                           weights=[self.word_vector_maxtrix],
                                           trainable=False,
                                           input_length=self.max_sent_len,
                                           name='embedding_layer_static')(input_layer)
            embed_layer = Concatenate()([embed_layer, embed_layer_static])
        # multi-filter
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embed_layer)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        concat_layer = Concatenate()(convs)
        dropout_layer = Dropout(0.5)(concat_layer)
        # classify
        output_layer = Dense(self.class_num, activation=self.last_activation)(dropout_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
    
    