# -*- coding: utf-8 -*-

"""
 lda模型训练
 利用sklearn框架进行lda模型训练
 模型训练整体思路：（主程序中为整体算法的训练思路）
 1、数据预处理，包括：切词、去停用词。
 2、利用CountVectorizer进行词频统计
 3、利用统计的词频进行词频向量构建
 4、lda模型训练

 训练可得lda两个分布：lda_model.transform(词频向量)——文档主题分布
                     lda_model.components_——单词主题分布

 利用CountVectorizer词频统计的单词映射表可以查看单词主题分布的具体单词

 calculate_similar类封装的是根据句向量进行句子之间余弦值计算以及文章之间相似度的计算
                 theta为计算余弦相似度时加入的避免分母为零

 代码中使用的数据集THUCNews 来源于http://thuctc.thunlp.org/
 Contributor：LinLin
 Reviewer：xionglongfei
"""


import pandas as pd
import numpy as np
import itertools
import pickle
import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def read_file(file_path):
    """
    将不同文件夹下的文档读到一起
    :param  file_path
    :return: dataframe
    """
    tmp = []
    folder_path = os.listdir(file_path)
    folder_list = [os.path.join(file_path, i) for i in folder_path]
    for i in folder_list:
        file_list = [os.path.join(i, a) for a in os.listdir(i)][:100]
        for file in file_list:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                a = f.read()
            tmp.append(a.replace('\n', ' '))
    dfx = pd.DataFrame(tmp, columns=['content'])
    return dfx


class calculate_similar():
    def __init__(self):
        self.theta = 0.0001
        self.stop_word = open('./stopwords.txt', encoding='utf-8')

    def cut_word(self, line):
        """
        :param line: 待分词的句子
        :return:针对文本进行分词并且去除停用词后的
        """
        word_list = jieba.lcut(line, cut_all=False, HMM=False)
        word_list = [i for i in word_list if i not in self.stop_word]
        word_line = ' '.join(word_list)
        return word_line

    def calculate_similarity(self, sen1, sen2):
        """
        计算两向量余弦相似度
        """
        fraction = np.dot(sen1, sen2)
        denominator = (np.linalg.norm(sen1) * (np.linalg.norm(sen2))) + self.theta
        return fraction / denominator

    def create_graph(self, doc_vec, aim_vec):
        """
        :param doc_vec:已经训练好的文档主题分布矩阵
        :param aim_vec:待推荐的文档主题分布
        :return:新增文章与原有文章的主题相似度
        """
        num = len(doc_vec)
        aim_mun = len(aim_vec)
        board = np.zeros([num, aim_mun])

        for i, j in itertools.product(range(num), range(aim_mun)):
            board[i][j] = self.calculate_similarity(doc_vec[i], aim_vec[j])

        return board


if __name__ == '__main__':
    data = read_file('E:/data/THUCNews')
    cs = calculate_similar()
    stop_words = [i.replace('\n', "") for i in open('./stopwords.txt', encoding='utf-8')]
    data['content'] = data.apply(lambda row: cs.cut_word(row['content']), axis=1)  # 对content内容进行切词处理
    count_Vector = CountVectorizer()  # 初始化CountVectorizer词频统计类
    cnt_model = count_Vector.fit(data['content'])  # 利用data的内容数据进行模型构建
    cnt_vector = cnt_model.transform(data['content'])  # 获得data数据的词频向量
    lda = LatentDirichletAllocation(n_topics=10,  # lda模型参数设置
                                    learning_offset=50,
                                    random_state=0,
                                    max_iter=30)
    lda_model = lda.fit(cnt_vector)  # lda模型训练
    doc_topic = lda_model.transform(cnt_vector)  # 文档主题分布
    word_topic2 = lda_model.components_  # 单词主题分布
    print(word_topic2)

    # 打印每个主题的前十个单词
    index_word = {value: key for key, value in cnt_model.vocabulary_.items()}
    ten_topic = [i.argsort()[-20:][::-1] for i in word_topic2]
    for i in ten_topic:
        temp = [index_word[a] for a in i]
        print(temp)
        print("---------")

    # 保存词频统计的模型确保下一次进行词频矩阵计算的时候词语的位置以及维度是一致的
    count_model_save = open('./model/count_vector_model.pkl', 'wb')
    pickle.dump(cnt_model, count_model_save)
    count_model_save.close()

    # 保存lda模型方便下一次直接使用
    lda_model_save = open('./model/lda_model.pkl', 'wb')
    pickle.dump(lda_model, lda_model_save)
    lda_model_save.close()

    test_data = read_file('E:/data/THUCNews')  # 新的文章
    test_data['content'] = test_data.apply(lambda row: cs.cut_word(row['content']), axis=1)  # 切词去停用词
    test_vector = cnt_model.transform(test_data['content'])  # 利用上面的cnt_model进行词频统计
    test_topic = lda_model.transform(test_vector)  # 利用训练好的lda模型对新的数据进行分布计算
    similarity = cs.create_graph(doc_topic, test_topic)  # 计算主题相似度
    recommend_indexes = np.argmax(similarity, axis=0)  # 获得新的文章与原来的主题相似度最高的文章的index列表
