#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
封装mysql操作的工具类
Contributor：PanYunSong
Reviewer：xionglongfei
"""


import pymysql
from config import rec_db_config


class MysqlDao(object):
    def __init__(self):
        self.connection = pymysql.connect(**rec_db_config)

    def __get_connect(self):
        temp_cur = self.connection.cursor()
        if not temp_cur:
            raise (NameError, "Mysql connection failed!")
        else:
            return temp_cur

    def execute_query(self, sql_str, sql_params=()):
        cur = None
        res = None
        try:
            cur = self.__get_connect()
            cur.execute(sql_str, sql_params)
            res = cur.fetchall()
        except Exception as e:
            raise e
        finally:
            if cur is not None:
                cur.close()
            return res

    def close_conn(self):
        if self.connection:
            self.connection.close()
