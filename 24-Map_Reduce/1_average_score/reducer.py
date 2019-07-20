#!/usr/bin/env python3

"""
平均成绩Reducer
Contributor：lisilang
Reviewer：xionglongfei
"""


import sys  

# 当前名称
current_name = None
# 当前得分
current_score = 0
# 科目数
subject_num = 0

for line in sys.stdin:
    # 把输入 k,v 切割为 [名称，科目，得分]
    line = line.strip("\n")
    name,subject,score = line.split(' ')
    try:
        # 得分转换为整数型
        score = int(score)
    except ValueError:
        continue
    # 当前名称与切割名称是否一致
    if current_name == name:
        # 得分累加，科目数累加1
        subject_num = subject_num + 1
        current_score = current_score + score
    else:
        if current_name:
            # 平均分 = 总分/科目数
            avg_score = current_score/(subject_num+1)
            # 姓名：current_name
            # 当前科目数：subject_num+1
            # 当前总分：current_score
            # 当前平均分：round(avg_score,2) 如果有小数，取小数点后2位
            print("姓名:", current_name, "\t科目数:", subject_num+1, "\t总分:", current_score, "\t平均分:", round(avg_score,2))
        # 更新当前名称
        current_name = name
        # 更新当前得分
        current_score = score
        # 重置当前科目数
        subject_num = 0
if current_name == name:
    # 平均分 = 总分/科目数
    avg_score = current_score / (subject_num + 1)
    # 姓名：current_name
    # 当前科目数：subject_num+1
    # 当前总分：current_score
    # 当前平均分：round(avg_score,2) 如果有小数，取小数点后2位
    print("姓名:", current_name, "\t科目数:", subject_num + 1, "\t总分:", current_score, "\t平均分:", round(avg_score, 2))
