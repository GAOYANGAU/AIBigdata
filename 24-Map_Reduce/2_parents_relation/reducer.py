#!/usr/bin/env python3 

"""
父子关系Reducer
Contributor：lisilang
Reviewer：xionglongfei
"""


import sys  
from collections import defaultdict

# 生成一个 value 仅有 list 的字典
name_dict = defaultdict(list)

for line in sys.stdin:
    line = line.strip("\n")
    key,value = line.split(' ')
    # 所有key 添加到字典key
    # value 累加到 valuelist
    name_dict[key].append(value)

for key,value in name_dict.items():
    # 只输出 value 大于2的 key,value
    if len(value) >= 2:
        print(key,"\t",value)




