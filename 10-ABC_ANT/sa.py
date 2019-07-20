"""
  模拟退火算法
  Contributor：hebinbin
  Reviewer：xionglongfei
  算法入口：SASystem.run()
  算法原理：
   模拟退火算法来源于固体退火原理，是一种基于概率的算法，将固体加温至充分高，再让其徐徐冷却，加温时，固体内部粒子随温升变为无序状，内能增大，而徐徐冷却时粒子渐趋有序，在每个温度都达到平衡态，最后在常温时达到基态，内能减为最小
  算法思路：
   当前温度temp := 最高温度T0 (从最高温度开始迭代)
   min_route := 保存当前最优路径
   cur_rout := 当前路径
   repeat
       for i := 1...N do
           new_rout = 在cur_rout周围随机搜索得到的新的路径
           if new_rout的总长度 < min_route的总长度 then
               min_route替换为new_rout
           else
               根据温度计算概率，并以该概率将min_route路径替换为new_rout
       end
       保存当前的温度搜索出的min_rout
       降温得到新temp
   until temp > 最小温度
"""

import math
import random
from tools import Pos, Node
from tools.draw import draw_map

data_file = './resources/wowmap/飞行点坐标.txt'
output_png = './outputs/sa_{dist}.jpg'  # 输出的路径图片
T0 = 1000  # 最高温度
T_MIN = 5  # 最小温度
LOOP_NUM_COFF = 5  # 指定温度下的迭代次数系数：节点数量的倍数
R_EARTH = 6400  # 地球半径 km (仅用于中国地图时，WOW地图无用)


class SASystem(object):
    """退火模拟算法"""

    def __init__(self):
        self.node_name2pos, self.node_list = SASystem.read_data_from_file(data_file)
        self.node_num = len(self.node_list)
        self.cur_temp = T0  # 当前温度
        self.loop_counter = 0
        self.loop_num = LOOP_NUM_COFF * len(self.node_list)  # 每个温度的迭代次数
        self.cur_route = [node for node in self.node_list]
        self.cur_min_distance = self.calc_path_distance(self.cur_route)  # 当前的最短距离

    @staticmethod
    def read_data_from_file(file_path):
        """"读取坐标数据"""
        node_map = {}
        node_list = []
        index = 0
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                if not line:
                    continue
                line_items = line.split()
                assert len(line_items) == 3, "地图坐标数据格式错误：%s" % file_path
                pos = line_items[2].split(",")
                name = line_items[1]
                log = float(pos[0])
                lat = float(pos[1])
                node_map.setdefault(name, (log, lat))
                node_list.append(Node(index, name, Pos(log, lat)))
                index += 1
        return node_map, node_list

    def update_temp(self):
        """更新温度"""
        self.cur_temp = T0 / self.loop_counter

    def distance_ij_accurate(self, node_i: Node, node_j: Node):
        """根据坐标计算几何直线距离"""
        pos_i = node_i.pos
        pos_j = node_j.pos
        return math.sqrt(math.pow(pos_i.log - pos_j.log, 2) + math.pow(pos_i.lat - pos_j.lat, 2))

    def distance_ij_accurate_lon_lat(self, node_i: Node, node_j: Node):
        """精确通过经纬度计算两地距离"""
        pos_i = node_i.pos
        pos_j = node_j.pos
        cos_theta = math.cos(math.pi * abs(pos_i.log - pos_j.log) / 180) + \
                    math.cos(math.pi * abs(pos_i.lat - pos_j.lat) / 180) - 1
        theta = math.acos(cos_theta)
        return theta * R_EARTH

    def calc_path_distance(self, node_list):
        """计算路径环路长度，需要加上第一个与最后一个的长度"""
        distance = 0
        for i in range(len(node_list)):
            j = i + 1 if i != len(node_list) - 1 else 0
            distance += self.distance_ij_accurate(node_list[i], node_list[j])
        return distance

    def judge_with_temp(self, new_dis):
        """根据当前温度计算转移概率"""
        p = math.exp(-(new_dis - self.cur_min_distance) / self.cur_temp)
        return random.random() < p

    def copy_and_random_route(self):
        """复制并产生一个新的数组"""
        copy_route = [node for node in self.cur_route]
        start_idx = random.randint(1, self.node_num - 1)
        end_idx = random.randint(1, self.node_num - 1)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        s = start_idx
        e = end_idx
        while s < e:
            copy_route[s], copy_route[e] = copy_route[e], copy_route[s]
            s += 1
            e -= 1
        return copy_route

    def run_loop(self):
        """每个温度下的迭代"""
        for counter in range(self.loop_num):
            new_route = self.copy_and_random_route()
            new_dis = self.calc_path_distance(new_route)
            if new_dis < self.cur_min_distance:
                self.cur_min_distance = new_dis
                self.cur_route = new_route
            elif self.judge_with_temp(new_dis):
                self.cur_min_distance = new_dis
                self.cur_route = new_route

    def run(self):
        """迭代降温"""
        while self.cur_temp >= T_MIN:
            self.loop_counter += 1
            self.run_loop()
            if self.loop_counter % 100 == 0:
                print("loop: {}, temp: {}, dist: {}".format(self.loop_counter, self.cur_temp, self.cur_min_distance))
            self.update_temp()
        route_indexes = [node.index for node in self.cur_route]
        route_indexes.append(route_indexes[0])
        return route_indexes


if __name__ == '__main__':
    sa = SASystem()
    min_route = sa.run()
    output_png = output_png.format(dist=round(sa.cur_min_distance, 3))
    draw_map(min_route, output_png)
    print("--------------------------------------------")
    print("best distance: %s" % (sa.cur_min_distance,))
    print("best path: %s" % min_route)
    print("output image: %s" % (output_png,))
