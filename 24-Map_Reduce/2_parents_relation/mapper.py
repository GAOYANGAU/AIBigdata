#!/usr/bin/env python3 

"""
父子关系Mapper
Contributor：lisilang
Reviewer：xionglongfei
"""
  
  
import sys  

for line in sys.stdin:
    '''
    输入值切割为：
        老张 小张
        小张 老张
    '''
    line = line.strip("\n")
    list = line.split(' ')
    key = list[0]
    value = list[1]
    print(key,value)
    print(value,key)
