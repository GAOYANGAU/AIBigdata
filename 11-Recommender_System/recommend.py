#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RecommendEngine新闻推荐引擎，有两个流程构成：
1 粗排：根据所有用户的点击行为，采用Item CF算法召回新闻，构成召回候选集。
2 精排：根据用户的点击行为，采用LDA的主题分布的余弦相似度对召回候选集做精排，精排的结果推送给用户
输入：用户ID
输出：文章ID
Contributor：PanYunSong
Reviewer：xionglongfei
"""


from resource import MYSQL_CLIENT
from news_sim_lda import NewsSimLda
from item_cf import get_click_action, item_similarity, item_cf_recommend


class RecommendEngine(object):
    def __init__(self, user_id, n_article=3):
        """
        :param user_id: 用户ID
        :param n_article:int, 推荐n_article新闻
        """
        self.user_id = str(user_id)
        self.click_data = None  # 用户点击行为数据
        self.recall_data = []  # 召回集数据
        self.recommend_newsid = []  # 给用户推荐的新闻ID
        self.n_article = n_article  # 给用户推n_article篇文章

    def _get_user_action(self):
        """
        从msyql获取用户的用户点击行为
        :return:
        """
        # 查用户点击新闻数据
        sql = "select newsid from user_click where userid=%s limit 1 " % (self.user_id)
        for item1 in MYSQL_CLIENT.execute_query(sql):
            newsid = item1["newsid"]
            # 查用户点击新闻的内容
            sql = "select content from article where newsid=%s;" % newsid
            for item2 in MYSQL_CLIENT.execute_query(sql):
                self.click_data = (item1["newsid"], item2["content"])

    def _recall(self):
        """
        召回模块
        :return:
        """
        mysql_of_actions = 'user_click'
        user_news = get_click_action(mysql_of_actions)
        # 获取召回集的文章ID
        data = item_cf_recommend(item_similarity(user_news), [self.user_id], user_news)
        for newsid in data[self.user_id]:
            # 获取召回集中文章的内容
            sql = "select content from article where newsid=%s;" % newsid
            for item in MYSQL_CLIENT.execute_query(sql):
                self.recall_data.append({'newsid': newsid, 'content': item["content"]})

    def _rank(self):
        """
        排序模块
        :return:
        """
        score_list = []
        # 遍历召回集中的新闻，求相似度
        for recall_data in self.recall_data:
            # 求两篇文章的相似度
            sim = NewsSimLda.calc_similar(recall_data['content'], self.click_data[-1])
            score_list.append((recall_data['newsid'], sim))

        # 排序模型，得分，从高分到低分
        score_list.sort(key=lambda x: x[-1], reverse=True)

        # 给用户推送self.n_article篇文章
        self.recommend_newsid = [newsid for newsid, _ in score_list[0:self.n_article]]

    def recommend_news(self):
        """
        给用户推荐新闻
        :return:
        """
        # 1获取用户行为
        self._get_user_action()
        # 2调用召回模型，完成粗排
        self._recall()
        # 3调研排序模型，完成精排
        self._rank()


if __name__ == '__main__':
    userid = 8  # 用户ID
    # 实例化推荐引擎实例
    recommendEngine = RecommendEngine(userid)
    # 给userid = 8推荐新闻
    recommendEngine.recommend_news()
    # 输出推荐文章的ID
    print(recommendEngine.recommend_newsid)
