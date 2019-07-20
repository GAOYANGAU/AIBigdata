#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jieba分词是一个很常用的分词工具，我们在这里对jieba
做一些简单的介绍和演示，更加详细的用法请参见jieba的官方文档：
https://github.com/fxsjy/jieba

Contributor：songchao
Reviewer：xionglongfei

jieba支持三种分词模式：

    精确模式，试图将句子最精确地切开，适合文本分析；
    全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
    搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。

支持繁体分词
支持自定义词典
"""


import jieba

sentence = '中国科学院大学是一所刚刚成立的大学。'

# 精确模式
words = jieba.cut(sentence, cut_all=False) 
print(' '.join(words))

# 全模式
words = jieba.cut(sentence, cut_all=True) 
print(' '.join(words))

# 搜索模式
words = jieba.cut_for_search(sentence) 
print(' '.join(words))


# 繁体
sentence2 = '中國科學院大學是一所剛剛成立的大學。'
# 精确模式
words = jieba.cut(sentence2, cut_all=False) 
print(' '.join(words))

# 全模式
words = jieba.cut(sentence2, cut_all=True) 
print(' '.join(words))

# 搜索模式
words = jieba.cut_for_search(sentence2) 
print(' '.join(words))

# 用户自定义词典
sentence3 = '机器学习是人工智能的重要组成部分。'

words = jieba.cut(sentence3) # 不加自定义词典
print(' '.join(words))

jieba.load_userdict('userdict.txt')
words = jieba.cut(sentence3) # 加自定义词典
print(' '.join(words))