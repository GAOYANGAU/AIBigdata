"""
  蚁群算法
  Contributor：hebinbin
  Reviewer：xionglongfei
  参考自：Colorni A, Dorigo M, Maniezzo V. Distributed optimization by ant colonies[C]//Proceedings of the first
           European conference on artificial life. 1991, 142: 134-142

   算法入口： AntSystem.run()
   算法原理：
    用蚂蚁的行走路径表示待优化问题的可行解，整个蚂蚁群体的所有路径构成待优化问题的解空间。路径较短的蚂蚁释放的信息素量较多，随着时间的推进，较短的路径上累积的信息素浓度逐渐增高，选择该路径的蚂蚁个数也愈来愈多。最终，整个蚂蚁会在正反馈的作用下集中到最佳的路径上。
   算法思路：
     初始化：
       确定城市数N
       初始化一个N*N的二维数组Trail，代表Trail[i][j] = Trail[j][i]代表节点i与节点j路径上的信息素浓度
       每个城市分配一定数量的蚂蚁，每只蚂蚁维护一个已访问节点路径列表
     for i := 1...迭代轮数 do
       repeat
           遍历所有蚂蚁，每只蚂蚁根据路径信息素浓度与距离计算概率，用轮赌算法选择下一个节点
       until 所有蚂蚁走完了所有节点
       保存当前一轮搜索到的最优路径
       根据当前一轮的信息素变化与蒸发率，更新信息素矩阵Trail
       保存当前最优的结果
"""

import math
import random
import datetime
from tools import Pos
from tools.draw import draw_map

data_file = './resources/wowmap/飞行点坐标.txt'
output_png = './outputs/ant_{dist}.jpg'  # 输出的路径图片
LOOP_NUM = 50  # 循环次数
ANT_COUNT_PER_NODE = 10  # 每个节点上分配的蚂蚁数量
ALPHA = 1  # 概率参数alpha
BELTA = 5  # 概率参数belta
Q = 100  # Q
P = 0.5  # 蒸发因子
# 使用中国地图参数，WOW地图时不需要用
R_EARTH = 6400


class Node(object):
    """
    节点，每个节点上初始化有一定数量的蚂蚁
    """

    def __init__(self, node_index, name, pos: Pos, antsys):
        self.antsys = antsys
        self.name = name  # 节点名称
        self.index = node_index  # 节点编号
        self.pos = pos  # 节点坐标 (x,y)
        # 每个节点上的蚂蚁列表
        self.ants = [Ant(node_index, ANT_COUNT_PER_NODE * node_index + ant_index, antsys)
                     for ant_index in range(ANT_COUNT_PER_NODE)]

    def init_data(self):
        """每新一轮计算前执行初始化"""
        for ant in self.ants:
            ant.init_data()


class Ant(object):
    """蚂蚁"""

    def __init__(self, init_node_index, ant_index, antsys):
        self.antsys = antsys
        self.index = ant_index  # 蚂蚁编号
        self.init_node_index = init_node_index  # 初始节点编号
        self.cur_node_index = init_node_index
        self.visited_nodes = []  # 访问过的节点编号列表
        self.unvisited_nodes = []  # 待访问的节点编号列表
        self.total_distance = 0  # 行程长度
        self.init_data()

    def init_data(self):
        """每轮计算前都需要初始化数据"""
        self.cur_node_index = self.init_node_index
        self.visited_nodes = [self.init_node_index]
        # 待访问节点编号列表
        self.unvisited_nodes = []
        for index in range(AntSystem.node_count):
            if index != self.init_node_index:
                self.unvisited_nodes.append(index)
        self.total_distance = 0

    def distance_ij(self, i, j):
        """根据经纬度计算距离"""
        return self.distance_ij_accurate(i, j)

    def distance_ij_accurate(self, i, j):
        """根据坐标计算几何直线距离"""
        pos_i = self.antsys.nodes_list[i].pos
        pos_j = self.antsys.nodes_list[j].pos
        return math.sqrt(math.pow(pos_i.log - pos_j.log, 2) + math.pow(pos_i.lat - pos_j.lat, 2))

    def distance_ij_accurate_log_lat(self, i, j):
        """精确通过经纬度计算两地距离"""
        pos_i = self.antsys.nodes_list[i].pos
        pos_j = self.antsys.nodes_list[j].pos
        cos_theta = math.cos(math.pi * abs(pos_i.log - pos_j.log) / 180) + math.cos(
            math.pi * abs(pos_i.lat - pos_j.lat) / 180) - 1
        theta = math.acos(cos_theta)
        return theta * R_EARTH

    def theta_ij(self, i, j):
        """计算概率公式中theta_ij的值，代表下一节点的吸引力"""
        distance_ij = self.distance_ij(i, j)
        if distance_ij <= 0:
            print(self.index, self.init_node_index, self.cur_node_index, i, j)
        return 1 / distance_ij

    def wheel_select(self, next_node_probabilities):
        """轮盘赌法选出下一个节点"""
        rand = random.random()
        index = 0
        total_prob = 0
        for prob in next_node_probabilities:
            total_prob += prob
            if total_prob > rand:
                break
            index += 1
        return index

    def update_to_next_node(self):
        """根据转移概率选择下一个节点"""
        if not self.unvisited_nodes:
            return True
        next_node_probabilities = []  # 下一节点的概率列表
        for next_node_index in self.unvisited_nodes:
            tao = self.antsys.trails_data[self.cur_node_index][next_node_index]
            thta = self.theta_ij(self.cur_node_index, next_node_index)
            next_node_probabilities.append(math.pow(tao, ALPHA) * math.pow(thta, BELTA))
        total_value = 0
        for val in next_node_probabilities:
            total_value += val
        for idx in range(len(next_node_probabilities)):
            val = next_node_probabilities[idx]
            next_node_probabilities[idx] = val / total_value
        select_index = self.wheel_select(next_node_probabilities)
        next_node_index = self.unvisited_nodes[select_index]
        self.total_distance += self.distance_ij(self.cur_node_index, next_node_index)
        self.visited_nodes.append(next_node_index)
        # self.visited_node_set.add(next_node_index)
        self.cur_node_index = next_node_index
        self.unvisited_nodes[select_index] = self.unvisited_nodes[-1]
        self.unvisited_nodes.pop()
        is_empty = len(self.unvisited_nodes) == 0
        if is_empty:  # 走完了所有节点需要加上回到源节点的距离
            self.visited_nodes.append(self.init_node_index)
            self.total_distance += self.distance_ij(self.cur_node_index, self.init_node_index)
        return is_empty


class AntSystem(object):
    """蚂蚁系统算法实现"""
    node_count = 0
    ant_count = 0

    def __init__(self):
        self.nodes_name2pos, self.nodes_list = self.read_data_from_file(data_file)
        AntSystem.node_count = len(self.nodes_name2pos)
        AntSystem.ant_count = ANT_COUNT_PER_NODE * self.node_count
        # 信息素初始矩阵
        self.trails_data = [[0.1 for _ in self.nodes_list] for _ in self.nodes_list]
        self.delta_trails_data = []
        # 初始化信息素每轮变更矩阵
        self.init_data()

    def init_data(self):
        """初始化信息素每轮变更的矩阵为0"""
        self.delta_trails_data = [[0 for node in self.nodes_list] for node in self.nodes_list]
        for node in self.nodes_list:
            node.init_data()

    def read_data_from_file(self, file_path):
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
                node_list.append(Node(index, name, Pos(log, lat), self))
                index += 1
        return node_map, node_list

    @staticmethod
    def now():
        return datetime.datetime.now().timestamp()

    def update_trails(self):
        """更新信息素"""
        # 所有蚂蚁导致的信息素变化
        for node in self.nodes_list:
            for ant in node.ants:
                for idx_i in range(len(ant.visited_nodes) - 1):
                    idx_j = idx_i + 1
                    node_i = ant.visited_nodes[idx_i]
                    node_j = ant.visited_nodes[idx_j]
                    delta_trail = Q / ant.total_distance
                    self.delta_trails_data[node_i][node_j] += delta_trail
                    self.delta_trails_data[node_j][node_i] += delta_trail
        # print(self.delta_trails_data)
        # 新浓度 = 蒸发 + 变化
        for i in range(self.node_count):
            for j in range(self.node_count):
                old_trail = self.trails_data[i][j]
                self.trails_data[i][j] = self.trails_data[j][i] = P * old_trail + self.delta_trails_data[i][j]
        # print(self.trails_data)

    def run_one_loop(self):
        """所有蚂蚁走完一轮：走完所有节点"""
        least_distance = 1 << 32
        best_ant = None
        self.init_data()
        count = 0
        while True:
            count += 1
            is_all_ants_finished = True
            # 所有蚂蚁走一节点
            for node in self.nodes_list:
                for ant in node.ants:
                    # 每只蚂蚁选择下一个node
                    is_empty = ant.update_to_next_node()
                    is_all_ants_finished = is_all_ants_finished and is_empty
                    if is_empty and least_distance > ant.total_distance:
                        best_ant = ant
                        least_distance = ant.total_distance

            if is_all_ants_finished:
                break
        # 一轮结束时更新信息素
        self.update_trails()
        return best_ant

    def run(self):
        """算法入口"""
        loop_count = 0
        min_route = None
        min_distance = None
        loop_ant_opt_distance_list = []
        start = AntSystem.now()
        while True:  # 多轮
            loop_count += 1
            ant = self.run_one_loop()
            loop_ant_opt_distance_list.append(ant.total_distance)
            # 第一次搜索或找到
            if min_distance is None or  ant.total_distance < min_distance:
                min_distance = ant.total_distance
                min_route = [self.nodes_list[node].index for node in ant.visited_nodes]

            if loop_count % 10 == 0:
                timedelta = AntSystem.now() - start
                print("loop: %s, distance: %s, time: %s" % (loop_count, ant.total_distance, timedelta))
                start = AntSystem.now()
            if loop_count >= LOOP_NUM:
                # print(loop_ant_opt_distance_list)
                return min_distance, min_route


if __name__ == '__main__':
    antsys = AntSystem()
    min_distance, min_route = antsys.run()
    output_png = output_png.format(dist=round(min_distance, 3))
    draw_map(min_route, output_png)
    print("--------------------------------------------")
    print("best distance: %s, path: %s" % (min_distance, min_route))
    print("output image: %s" % (output_png,))
