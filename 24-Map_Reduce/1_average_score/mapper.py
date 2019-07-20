#!/usr/bin/env python3

"""
平均成绩Mapper
Contributor：lisilang
Reviewer：xionglongfei
"""


import sys  

for line in sys.stdin:
    '''
    输入值切割为：
        key: 名称
        value: 科目 得分
    '''
    line = line.strip("\n")
    list = line.split(' ')
    key = list[0]
    value = list[1]+" "+list[2]
    print(key,value)
