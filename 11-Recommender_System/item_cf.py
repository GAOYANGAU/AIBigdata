#!/usr/bin/env python
# coding=utf-8

"""
文章与文章之间的协同过滤：
1 从点击行为表中获取用户对文章的点击记录
2 利用点击记录生成文章与文章之间的相似度矩阵W
3 根据W来为每个用户生成指定数量的推荐文章集合
Contributor：chenlang
Reviewer：xionglongfei
"""


import math
import operator
import pandas as pd
import pymysql
from collections import defaultdict

rec_db_config = {
    'cursorclass': pymysql.cursors.DictCursor,
    'host': '10.13.133.161',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'db': 'recsys',
    'autocommit': False
}


def get_click_action(table_of_actions):
    """
    从点击行为表中获取点击行为记录
    :param table_of_actions: 点击行为表
    :return: (用户id,文章id)
    """
    click_sql = "select CAST(userid AS char) userid, CAST(newsid AS char) newsid, click_date from {action_table} "
    click_sql = click_sql.format(action_table=table_of_actions)
    db = pymysql.connect(**rec_db_config)
    action_values = pd.read_sql(click_sql, con=db)  # 写入pandas
    action_values.drop_duplicates(inplace=True)  # 去重
    user_news_pd = action_values[['userid', 'newsid']]
    return user_news_pd


def item_similarity(user_new_pd):
    """
    求文章与文章之间的相似度
    :param user_new_pd: (用户id,文章id)
    :return: 文章与文章之间的相似度
    """
    article_lengthOfusers = user_new_pd.groupby('newsid').apply(
        lambda x: pd.Series({'user_length': len(set(x['userid']))}))
    user_lengthOfnews = user_new_pd.groupby('userid'). \
        apply(lambda x: pd.Series({'article_length': len(set(x['newsid'])), 'article_list': list(set(x['newsid']))}))
    # 过滤出一个用户点击多条文章的记录，用来计算W
    user_lengthOfnews_legal = user_lengthOfnews[user_lengthOfnews['article_length'] > 1]

    dict_similar = defaultdict(dict)

    for row in user_lengthOfnews_legal['article_list']:
        for newsid_0 in row:
            if newsid_0 not in dict_similar:
                dict_similar[newsid_0] = defaultdict(float)
        for newsid_0 in row:
            for newsid_1 in row:
                if newsid_0 != newsid_1:
                    dict_similar[newsid_0][newsid_1] += 1
    for newsid_0, newsid_dict in dict_similar.items():
        for newsid_1, cnt in newsid_dict.items():
            dict_similar[newsid_0][newsid_1] = cnt / math.sqrt(
                article_lengthOfusers['user_length'][newsid_0] * article_lengthOfusers['user_length'][newsid_1])
    return dict_similar


def item_cf_recommend(W_dict, candi_users, user_news):
    """
    item_cf的用户推荐
    :param W_dict: 文章与文章之间的相似度
    :param candi_users: 有点击行为的用户集合
    :param user_news: (用户id,文章id)
    :return: item_cf的用户推荐
    """
    user_recommendOfarticles = dict()
    news_rank = dict()
    top_k = 5
    for user in candi_users:
        # 求user的点击文章set
        user_clk_items = set(user_news[user_news["userid"] == user]["newsid"])
        for clk_new in user_clk_items:
            # 求user的推荐集合
            recom_news = sorted(W_dict[clk_new].items(), key=operator.itemgetter(1), reverse=True)
            for item, score in recom_news:
                if item not in user_clk_items:
                    if item in news_rank:
                        news_rank[item] += score
                    else:
                        news_rank[item] = score
        recommend_news = [x[0] for x in sorted(news_rank.items(), key=operator.itemgetter(1), reverse=True)]
        recommend_news_pure = [i for i in recommend_news if i not in user_clk_items]  # 去掉已经点击的文章
        user_recommendOfarticles[user] = recommend_news_pure[:top_k]  # 为每个用户选取前top_k的文章集合
    return user_recommendOfarticles


if __name__ == "__main__":
    mysql_of_articles = 'article'    # 文章表
    mysql_of_actions = 'user_click'  # 点击行为表

    user_news = get_click_action(mysql_of_actions)  # 从点击行为表中获取点击行为记录
    W_dict = item_similarity(user_news)  # 求文章与文章之间的相似度
    print("ITEM_CF W:")
    print(W_dict)

    candi_users = list(set(user_news['userid'].values))  # 有点击行为的用户集合
    user_recommendOfarticles = item_cf_recommend(W_dict, candi_users, user_news)  # item_cf的用户推荐
    print('user_recommendOfarticles: ')
    print(user_recommendOfarticles)
