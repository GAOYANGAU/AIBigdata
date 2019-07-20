#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py
Contributor:PanYunSong
Reviewer:xionglongfei
"""


import pymysql
rec_db_config = {
    'cursorclass': pymysql.cursors.DictCursor,
    'host': '10.13.133.161',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'db': 'recsys',
    'autocommit': False
}
