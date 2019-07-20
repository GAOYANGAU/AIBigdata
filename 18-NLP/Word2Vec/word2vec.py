#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本算法适用于训练词向量
Contributor：songchao
Reviewer：xionglongfei
"""


import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten
from keras.layers.merge import dot

class Word2Vec(object):
    """
    Implementation of word2vec model by Keras, which only use 
    skip-gram and negative sampling.
    """
    def __init__(self, vec_dims, 
                 vocab_size=None,
                 window_size=3,
                 negative_samples=5.,
                 optimizer='rmsprop',
                 loss='binary_crossentropy',
                 batch_size=128,
                 epochs=10):
        self.vocab_size = vocab_size
        self.vec_dims = vec_dims
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.token = Tokenizer()
    
    def skip_gram(self, sequences):
        """
        Generates skipgram word pairs.
        
        Parameters:
            sequences: A word sequence (sentence), encoded as a list
                       of word indices (integers).
        Return:
            a sequence of word indexes (list of integers) into tuples of words of the form:
                - (word, word in the same window), with label 1 (positive samples).
                - (word, random word from the vocabulary), with label 0 (negative samples).
        """
        bi_tuple = []
        labels = []
        for seq in sequences:
            b, l = sequence.skipgrams(seq, 
                                      vocabulary_size=self.vocab_size,
                                      window_size=self.window_size,
                                      negative_samples=self.negative_samples)
            bi_tuple.extend(b)
            labels.extend(l)
        return np.asarray(bi_tuple), np.asarray(labels)
    
    def word2vec(self):
        center_word = Input(shape=(1, ), name='center_word')
        context_word = Input(shape=(1, ), name='context_word')
        
        center_embedding = Embedding(self.vocab_size+1,
                                     self.vec_dims,
                                     input_length=1,
                                     name='center_embedding')(center_word)
        context_embedding = Embedding(self.vocab_size+1,
                                      self.vec_dims,
                                      input_length=1,
                                      name='context_embedding')(context_word)
        
        merge_layer = dot([center_embedding, context_embedding],
                    axes=-1, name='dot')
        merge_layer = Flatten()(merge_layer)
        
        output = Dense(1, activation='sigmoid', name='output_layer')(merge_layer)
        self.model = Model(inputs=[center_word, context_word],
                           outputs=output)
    
    def train(self, corpus, save_path):
        self.token.fit_on_texts(corpus)
        if self.vocab_size is None:
            self.vocab_size = len(self.token.word_index)
        seqs = self.token.texts_to_sequences(corpus)
        couples, labels = self.skip_gram(seqs)
        print('couples shape: ', couples.shape)
        print('lables shape: ', labels.shape)
        cent_words = couples[:, :1]
        cont_words = couples[:, 1:]
        self.word2vec()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit([cent_words, cont_words], labels,
                       batch_size=self.batch_size,
                       epochs=self.epochs)
        # save weights
        with open(save_path, 'w+', encoding='utf8') as f:
            weights = self.model.get_weights()[0]
            f.write('%s %s' % (self.vocab_size, self.vec_dims))
            for w in self.token.word_index:
                if self.token.word_index[w] <= self.vocab_size:
                    f.write(w+' ')
                    f.write(' '.join(map(str, weights[self.token.word_index[w], :])))
                    f.write('\n')
                   
                    
if __name__ == '__main__':
    # load data from nltk
    from nltk.corpus import brown
    sentences = brown.sents()[:5000]
    # train
    word2vec = Word2Vec(vec_dims=100)
    word_vec = word2vec.train(sentences, save_path='vectors.txt')
    
    
                    