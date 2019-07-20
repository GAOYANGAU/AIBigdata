"""
Hidden Markov Model聚类
Contributor：chenronghua
Reviewer：xionglongfei
"""

import numpy as np

#初始化数据
#假设数组jin为"jin" 对应的汉字
jin = ['近', '斤', '今', '金', '尽']
#假设jin_per 表示"jin"是对应jin数组对应位置汉字可能的概率
jin_per = [0.3, 0.2, 0.1, 0.06, 0.03]

#假设 jintian数组表示”tian“对应的汉字
jintian = ['天', '填', '田', '甜', '添']
#假设 jintian_per数组表示"jintian"这两个拼音的对应汉字的条件概率
jintian_per = [
    [0.001, 0.001, 0.001, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.001, 0.001],
    [0.990, 0.001, 0.001, 0.001, 0.001],
    [0.002, 0.001, 0.850, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.001, 0.001]
]

#同上
wo = ['我', '窝', '喔', '握', '卧']
wo_per = [0.400, 0.150, 0.090, 0.050, 0.030]

#同上
women = ['们', '门', '闷', '焖', '扪']
women_per = [
    [0.970, 0.001, 0.003, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.001, 0.001],
    [0.001, 0.001, 0.001, 0.001, 0.001]
]


#获取某个词拼音的前N个最有可能的输出
def topn_from_ownword(oneword_per, N=5):
    index = []
    values = []
    oneword_per = np.array(oneword_per)
    #概率进行从小到大的升序排序， 并取最后N个
    for v in np.argsort(oneword_per)[::-1][:N]:
        index.append(v)
        values.append(oneword_per[v])
    return index, values

#获取一个词组拼音的前N个最有可能的输出
def topn_from_twoword(oneword_per, twoword_per, N=5):
    last = 0
    #前一个拼音对应汉字的个数
    word_num = len(oneword_per)

    for i in range(word_num):
        #当一个词组，例如当"jintian"，这一对词组，第一个拼音”jin“取”今“, 第二个拼音”tian“取天时的条件概率
        current = np.multiply(oneword_per[i], twoword_per[i])
        if i == 0:
            last = current
        else:
            #将计算出来的概率追加到结果列表中
            last = np.concatenate((last, current), axis=0)

    index = []
    values = []
    for v in np.argsort(last)[::-1][:N]:
        #计算概率高的词组对应的位置，v/word_num代表第一个汉字在第一个汉字中的位置， v%word_numv代表第二个汉字在第二个汉字中的位置
        index.append([int(v/word_num), v%word_num])
        #last[v]表示概率
        values.append(last[v])
    return index, values

#推理预测
def predict(word):
    N = 5
    if word == 'jin':
        indexs, _ = topn_from_ownword(jin_per, N)
        for i in indexs:
            print (jin[i])
            
    elif word == 'jintian':
        indexs, _ = topn_from_twoword(jin_per, jintian_per, N)
        for first_index, second_index in indexs:
            print (jin[first_index] + jintian[second_index])
            
    elif word == 'wo':
        indexs, _ = topn_from_ownword(wo_per)
        for i in indexs:
            print (wo[i])
            
    elif word == 'women':
        indexs, _ = topn_from_twoword(wo_per, women_per, N)
        for first_index, second_index in indexs:
            print (wo[first_index] + women[second_index])
            
    elif word == 'jintianwo':
        
        index1, values1 = topn_from_ownword(wo_per, N)
        index2, values2 = topn_from_twoword(jin_per, jintian_per, N)
        last = np.multiply(values1, values2)
        for i in np.argsort(last)[::-1][:N]:
            print (jin[index2[i][0]] + jintian[index2[i][1]] + wo[i])
            
    elif word == 'jintianwomen':
        index1, values1 = topn_from_twoword(jin_per, jintian_per, N)
        index2, values2 = topn_from_twoword(wo_per, women_per, N)
        last = np.multiply(values1, values2)
        for i in np.argsort(last)[::-1][:N]:
            print(jin[index1[i][0]] + jintian[index1[i][1]] + wo[index2[i][0]] + women[index2[i][1]])
    else:
        pass


if __name__ == '__main__':
    # 近
    # 斤
    # 今
    # 金
    # 尽
    predict('jin')  
    print("=====================================") 
    #今天
    #金田
    #近天
    #近填
    #近田
    predict('jintian')
    print("=====================================") 
    #我
    #窝
    #喔
    #握
    #卧
    predict('wo')
    # 我们
    # 我闷
    # 我门
    # 我焖
    # 我扪
    predict('women')
    print("=====================================") 
    # 今天我
    # 金田窝
    # 近天喔
    # 近填握
    # 近田卧
    predict('jintianwo')
    print("=====================================") 
    # 今天我们
    # 金田我闷
    # 近田我扪
    # 近填我焖
    # 近天我门  
    predict('jintianwomen')
    print("=====================================") 

