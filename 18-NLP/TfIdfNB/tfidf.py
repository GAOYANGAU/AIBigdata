#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用tf-idf对文本进行特征提取
Contributor：songchao
Reviewer：xionglongfei
"""


import os
import json
from collections import Counter
import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import keras.backend as K


class Preprocess(object):
    """
    Preprocessing data.
    """
    def __init__(self):
        self.label_index = {}
        self.doc_index = {}
        with open('data/word_index.json', encoding='utf8') as f:
            self.word_index = json.load(f)
        
    def load_data(self, data_path):
        """
        Load raw data.
        
        Parameters:
            data_path: str, raw data file path.
            
        Return:
            texts: list of texts;
            labels: list of labels;
            docs: list of document's names
        """
        texts = []
        labels = []
        docs = []
        for i, name in enumerate(sorted(os.listdir(data_path))):
            path = os.path.join(data_path, name)
            if os.path.isdir(path):
                self.label_index[name] = i
                for j, fname in enumerate(sorted(os.listdir(path))):
                    if fname.isdigit():
                        self.doc_index[fname] = j
                        fpath = os.path.join(path, fname)
                        with open(fpath, encoding='latin-1') as f:
                            texts.append(f.read())
                            labels.append(name)
                            docs.append(fname)
        return texts, labels, docs
            
    def counts(self, texts, docs):
        """
        Count word frequences, and convert the document name into index.
        
        Parameters:
            texts: list of texts;
            docs: list of documents.
            
        Return:
            tuple of word tokens, word counts and document tokens.
        """
        words_token = []
        docs_token = []
        words_cont = []
        for i, text in enumerate(texts):
            word_list = text.split()
            word_count = Counter(word_list)
            j = 0
            for w in word_count.keys():
                if w in self.word_index:
                    words_token.append(self.word_index[w])
                    words_cont.append(word_count[w])
                    j += 1
            docs_token.extend([self.doc_index[docs[i]]] * j)
        return np.array(words_token), np.array(words_cont), np.array(docs_token)
    
    
class TfIdf(object):
    """
    Implementation of Tf-Idf algorithm by scikit-learn.
    """
    def __init__(self):
        self.classifier = MultinomialNB()
    
    def sparse_matrix(self, word_index, doc_index, word_count, matrix_shape=None):
        """
        Creating sparse matrix with maximum dimension, to avoid dimension miss-match error.
        
        Parameters:
            word_index: word token array;
            doc_index: document token array;
            word_count: word count array.
            
        Return:
            Co-occurance Matrix.
        """
        co_matrix = sp.coo_matrix((word_count, (doc_index, word_count)), matrix_shape)
        co_matrix = co_matrix.tocsr()
        return co_matrix
    
    def tfidf(self, co_matrix):
        """
        Conveting the sparse matrix of test data into TFIDF values.
        
        Parameters:
            co_matrix: numpy.array
            
        Return:
            result of Tf-Idf.
        """
        tfidf = TfidfTransformer()
        tfidf_res = tfidf.fit_transform(co_matrix)
        return tfidf_res
    
    def MultiNB(self, x_train, y_train, x_test=None, y_test=None, mode='train'):
        """
        Text classification based on Tf-Idf using Multinomial Naive Bayes.
        
        Parameters:
            x_train: numpy array, training data
            y_train: numpy array, labels data
            x_test: numpy array, test data
            y_test: numpy array, test labels data
            mode: train mode or predict mode
                
        Return:
            prediction result, numpy array.
        """
        if mode == 'train':
            # train
            self.classifier.fit(x_train, y_train)
            # predict
            pred = self.classifier.predict(x_train[:100])
        if x_test is not None and y_test is not None:
            pred = self.classifier.predict(x_test)
            # Calculating accuracy score
            acc = accuracy_score(y_test, pred)
            print('Accuracy score: ', acc)
        return pred
    
    
    
if __name__ == '__main__':
    preprocess = Preprocess()
    tfidf = TfIdf()
    texts, labels, docs = preprocess.load_data('data/20_newsgroup/')
    label_array = []
    for l in labels:
        label_array.append(preprocess.label_index[l])
    label_array = np.array(label_array)
    words_token, words_cont, docs_token = preprocess.counts(texts, docs)
    co_maxtrix = tfidf.sparse_matrix(words_token, words_cont, docs_token, matrix_shape=(len(texts), words_token.shape[0]))
    tfidf_matrix = tfidf.tfidf(co_maxtrix)
    pred = tfidf.MultiNB(tfidf_matrix, label_array)
    print(pred)