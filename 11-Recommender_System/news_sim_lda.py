#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
利用已经训练好的LDA模型，求两篇文章的相似度
输入：两篇文章的内容
输出：两篇文章的相似度，取值返回[0,1]
思路：1、对给定的文章进行切词
     2、根据训练时构建的词表对切完词的文章向量化
     3、利用训练好的lda模型求出两篇文章的主题分布
     4、根据两篇文章的主题分布计算文章的相似度
Contributor：PanYunSong
Reviewer：xionglongfei
"""


import jieba
import numpy as np
import pandas as pd
from resource import COUNT_MODEL, LDA_MODEL, STOP_WORDS


class NewsSimLda(object):
    count_model = COUNT_MODEL  # CountVectorizer模型，词频统计模型
    lda_model = LDA_MODEL  # LDA模型
    theta = 0.0001
    stop_word = STOP_WORDS  # 停用词

    def __init__(self):
        pass

    @staticmethod
    def _cut_word(line):
        """
        :param line: 待分词的句子
        :return:针对文本进行分词并且去除停用词后的
        """
        word_list = jieba.lcut(line, cut_all=False, HMM=False)
        word_list = [i for i in word_list if i not in NewsSimLda.stop_word]
        word_line = ' '.join(word_list)
        return word_line

    @staticmethod
    def _calc_similarity_vec(sen1, sen2):
        """
        计算两向量余弦相似度
        """
        fraction = np.dot(sen1, sen2)
        denominator = (np.linalg.norm(sen1) * (np.linalg.norm(sen2))) + NewsSimLda.theta
        return fraction / denominator

    @staticmethod
    def calc_similar(string1, string2):
        str1 = NewsSimLda._cut_word(string1)  # 对两篇文章进行切词
        str2 = NewsSimLda._cut_word(string2)
        str1_vec = NewsSimLda.count_model.transform(pd.Series(str1))  # 根据原有的词频统计模型进行词频统计
        str2_vec = NewsSimLda.count_model.transform(pd.Series(str2))
        str1_topic = NewsSimLda.lda_model.transform(str1_vec)  # 利用训练好的lda模型求文章的主题分布
        str2_topic = NewsSimLda.lda_model.transform(str2_vec)
        similar = NewsSimLda._calc_similarity_vec(str1_topic[0], str2_topic[0])  # 计算两个分布的相似度

        return similar


if __name__ == "__main__":
    inst = NewsSimLda()
    string1 = str(
        """留学提醒：教育部发布赴比利时留学预警 针对最近我国赴比利时留学人员接连发生入境时被比海关拒绝或者办理身份证明时被比警方要求限期出境的事件，教育部23日提醒赴比利时留学人员应注意严格遵守比方相关规定。 据记者了解，发生以上问题的主要原因是：部分留学人员未能按大学或者语言培训中心录取通知书规定的时间报到，在入境时被比海关扣留，一旦学校答复不予注册，就被拒绝入境；有的留学人员听信网上发布的信息或传言，花钱购买所谓“合法”经济担保，办理身份证明；有的甚至使用假经济担保办理身份证明，比政府有关部门发现查证后，留学人员被要求限期离境。 为防止类似事件再次发生，教育部提醒赴比利时高校或者语言培训中心学习的留学人员，必须严格遵守比利时的相关规定，要按照通知的入学注册日期到学校报到。如果因故延迟，请事先与学校联系并获得批准。另外，不要轻信网上或者其他人发布的可以“有偿提供合法经济担保或合法身份证明”的信息，以免遭受不必要的损失。  (记者 张宗堂 熊聪茹）""")
    string2 = str("""奥运女足四强开奖：美国夺冠 头奖111注每注9千9 奥运会女足四强竞猜开奖：一等奖111注，每注奖金9964元，女足四强竞猜销售量为1701622元。男足四强竞猜销售量3002340元。""")
    similar = inst.calc_similar(string1, string2)
    print(similar)
