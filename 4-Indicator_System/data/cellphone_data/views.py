from django.shortcuts import render
import json
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:password@localhost/test_database", encoding='utf-8')    #正式环境


def echarts(request):
    """
    接收前端页面请求，并将处理好的数据返回前端页面
    :param request:
    :return:
    """
    res = dict()
    if request.method == "POST":
        score_type = request.POST.get("score_name")
        city, number, time, option = get_data(score_type)
        res = {"city": city, "number": number, "time": time, "option": option}
        print(res)
    return render(request, "echarts.html", {"data": json.dumps(res)})


def get_data(score_type):
    """
    根据不同的请求，返回数据
    :param score_type: 请求数据类型
    :return:
    """
    if score_type == "time_rollup":
        return time_rollup()
    if score_type == "time_drilldown":
        return time_drilldown()
    if score_type == "location_rollup":
        return location_rollup()
    if score_type == "location_drilldown":
        return location_drilldown()


def time_rollup():
    """
    提取数据库中的数据，按时间纬度上卷
    :return:
    """
    city = list()
    number = list()
    time = list()
    option = "按时间上卷"
    sql = "select * from Cellphone_Data"
    res = engine.execute(sql)
    for i in res:
        if list(i)[2] in city:
            number[city.index(list(i)[2])] += 1
        else:
            city.append(list(i)[2])
            number.append(1)
        if list(i)[3] not in time:
            time.append(list(i)[3])

    return city, [number], time, option


def time_drilldown():
    """
    提取数据库中的数据，按时间纬度下钻
    :return:
    """
    number, city = quater_count()
    time = ['第一季度', '第二季度', '第三季度', '第四季度']
    option = "按时间下钻"

    return city, number, time, option


def location_rollup():
    """
    提取数据库中的数据，按地点纬度上卷
    :return:
    """
    city = list()
    number = list()
    time = list()
    option = "按销售地上卷"
    sql = "select * from Cellphone_Data"
    res = engine.execute(sql)
    for i in res:
        if list(i)[1] in city:
            number[city.index(list(i)[1])] += 1
        else:
            city.append(list(i)[1])
            number.append(1)
        if list(i)[3] not in time:
            time.append(list(i)[3])

    return city, [number], time, option


def location_drilldown():
    """
    提取数据库中的数据，按时间纬度下钻
    :return:
    """
    city = list()
    number = list()
    time = list()
    option = "按销售地下钻"
    sql = "select * from Cellphone_Data"
    res = engine.execute(sql)
    for i in res:
        if list(i)[2] in city:
            number[city.index(list(i)[2])] += 1
        else:
            city.append(list(i)[2])
            number.append(1)
        if list(i)[3] not in time:
            time.append(list(i)[3])

    return city, [number], time, option


def quater_count():
    """
    对各城市季度数据进行统计
    :return:
    """
    quater_temp = list()
    city = list()
    quater_data = list()
    j = 0
    for n in range(4):
        sql = "select * from Cellphone_Data where month between {} and {}".format(j+1, j+3)
        print(sql)
        res = engine.execute(sql)
        for i in res:
            if list(i)[2] in city:
                quater_temp[city.index(list(i)[2])] += 1
            else:
                city.append(list(i)[2])
                quater_temp.append(1)
        quater_data.append(quater_temp)
        quater_temp = [0] * len(city)
        j += 3

    return quater_data, city

