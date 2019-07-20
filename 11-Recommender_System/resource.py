#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
资源文件加载
Contributor：PanYunSong
Reviewer：xionglongfei
"""


import pickle
from mysql_dao import MysqlDao

COUNT_MODEL = pickle.load(open('./model/count_vector_model.pkl', 'rb'))
LDA_MODEL = pickle.load(open('./model/lda_model.pkl', 'rb'))
STOP_WORDS = open('./stopwords.txt', encoding='utf-8')
MYSQL_CLIENT = MysqlDao()
