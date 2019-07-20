# -*- coding:UTF-8 -*-

"""
创建数据，并将数据插入数据库
Contributor：zhangjiarui
Reviewer：xionglongfei
"""


import random
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from create_database import Cellphone_Data
import json, pymysql
import time

engine = create_engine("mysql+pymysql://root:password@localhost/test_database", encoding='utf-8')      #本地

# 数据源列表
province_list = ["广东", "四川", "陕西", "湖北"]
city_list = [["深圳", "广州"], ["成都", "绵阳", "西昌"], ["西安", "汉中"], ["武汉", "荆州"]]

def create_data(num):
    """
    创建数据
    :param num:创建数据条数
    :return:
    """
    cellphone_data = list()
    for i in range(num):
        cellphone_info = dict()
        cellphone_info['province'] = random.sample(province_list, 1)[0]
        cellphone_info['city'] = random.sample(city_list[province_list.index(cellphone_info['province'])], 1)[0]
        # cellphone_info['brand'] = random.sample(brand_list, 1)[0]
        # cellphone_info['cellphone_model'] = random.sample(cellphone_model_list, 1)[0]
        cellphone_info['year'] = str(2017)
        cellphone_info['month'] = str(random.randint(1, 12))
        cellphone_data.append(cellphone_info)

    return cellphone_data


def insert_data(data):
    """
    将数据插入数据库
    :param data: 创建的数据
    :return:
    """
    engine.execute(
        Cellphone_Data.__table__.insert(),
        data
    )

if __name__ == '__main__':
    cellphone_data = create_data(100)
    insert_data(cellphone_data)
