"""
  ABC artificial bee colony 人工蜂群算法
  Contributor：hebinbin
  Reviewer：xionglongfei
  算法参考自文献：
  Pathak N, Tiwari S P. Travelling salesman problem using bee colony with SPV[J]. organization, 2012, 13: 18.
  雇佣蜂：Employed Bees, 跟随蜂：Onlooker Bees, 侦查蜂：Scout Bees
  算法原理：
  食物源（这里有Solution表示）的质量fitness用TSP环路路径长度代表，路径越短，质量越高（注：下面的算法用距离的倒数计算fitness，为了扩大距离间的差别，对倒数进行了求幂）；
   雇佣蜂数量与食物源数量一致，搜索食物源信息
   跟随蜂的数量与雇佣蜂一样，舞蹈区等待雇佣蜂的信息，并根据食物源的质量计算概率选择一个食物源去搜索
   因蜂群算法一般用于计算连续问题，而TSP问题是离散问题，所以在文献中，作者做了特殊处理：
       1. 每个食物源会有一个D维数组Td，D维对应城市节点数量，Td数组中的值为连续值，初始为随机数
       2.为了将连续值的Td转换为离散的节点序号，作者做了SPV转换，就是将Td数组中值的大小排名作为城市的序号，将Td转换为了Sd整数数组，Sd中的值即代表城市的序号
   当计算一定轮数后，如果雇佣蜂与跟随蜂均没有在食物源周围找到更优的解，则雇佣蜂会转换侦查蜂，开始随机搜索新的食物源（即将该食物源的Td随机重新初始化）
   雇佣蜂: 搜索食物源，将获得食物源的质量
   跟随蜂: 根据食物源的质量计算概率，并以概率选择前往一个食物源，在该食物源周围搜索新食物源，找到更优的食物源则替换当前食物源（寻找局部最优解）
   侦查蜂：当一定轮数还末找到更优的解，则雇佣蜂转换为侦查蜂，随机寻找新食物源（全局搜索能力）
  算法流程：
   随机初始化NF个食物源(Solution)的D维Td、Sd数组
   repeat
      NF只雇佣蜂前往对应的Solution进行局部搜索
      NF只跟随蜂根据雇佣峰获得的每个Solution的fitness，计算概率，并以轮赌算法选出一个Solution前去执行局部搜索
      侦查蜂随机搜索
   until 最大轮数
"""

import math
import random
from tools import Pos, Node
from tools.draw import draw_map


data_file = './resources/wowmap/飞行点坐标.txt'
output_png = './outputs/abc_{dist}.jpg'  # 输出的路径图片
# 蜜源的数量
NF = 50
# 进行多少轮迭代后蜜源耗尽，耗尽前还末让到更优的蜜源则雇佣蜂变为侦查蜂
FOOD_EXHAUSTED_LOOP_NUM = 100
# 计算多少轮后退出循环
MAX_LOOP_NUM = 1000
# 中国地图时用，WOW地图不用
R_EARTH = 6400


class Solution(object):
    """食物/蜜源，也即solution"""
    def __init__(self, index, nodes_list):
        self.index = index
        self.nodes_list = nodes_list
        self.city_num = len(nodes_list)
        self.td = [self.random_t() for _ in nodes_list]
        self.fitness = None
        self.distance = 0  # 当前环路距离
        self.sd = Solution.update_sd_from_td(self.city_num, self.td)  # SPV转换Td后的数组
        new_fitness, new_distance = Solution.calc_fitness_of_sd(self.sd, self.nodes_list)
        self.update_best_fitness(new_fitness, self.td, self.sd, new_distance)
        self.search_counter =  0  # 记录搜索次数，如果超过限制还末找到更优的值则丢弃该蜜源

    @staticmethod
    def update_sd_from_td(city_num, td: list):
        """
        对Td执行SPV转换到Sd，转换规则：按递增排序后的序号转换
        如，Td=[5, 1.5, 0, 2, -0.5]，SPV转换得Sd=[4, 2, 1, 3, 0]
        计算过程：先排序，再执行二分查找
        """
        sd = [None for _ in range(city_num)]
        sorted_td = sorted(td)
        labels = [False for _ in range(city_num)]
        for idx in range(city_num):
            val = td[idx]
            sorted_idx = Solution.binary_find_index_in_array(val, sorted_td, 0, len(sorted_td) - 1)
            # 找出的索引可能有重复值，通过lables数组标记辅助计算，找到最小的末被标记的索引
            least_idx = Solution.find_min_index_in_lable_array(sorted_idx, sorted_td, labels)
            sd[idx] = least_idx
        return sd

    @staticmethod
    def distance_ij_accurate_log_lat(node_i: Node, node_j: Node):
        """精确通过经纬度计算两地距离"""
        cos_theta = math.cos(math.pi * abs(node_i.pos.log - node_j.pos.log) / 180) + \
                    math.cos(math.pi * abs(node_i.pos.lat - node_j.pos.lat) / 180) - 1
        theta = math.acos(cos_theta)
        return theta * R_EARTH

    @staticmethod
    def distance_ij_accurate(node_i: Node, node_j: Node):
        """通过坐标计算几何直线距离"""
        dx2 = math.pow((node_i.pos.log - node_j.pos.log), 2)
        dy2 = math.pow((node_i.pos.lat - node_j.pos.lat), 2)
        return math.sqrt(dx2 + dy2)

    @staticmethod
    def calc_path_distance(node_list):
        """计算路径环路长度，需要加上第一个与最后一个的长度"""
        distance = 0
        for i in range(len(node_list)):
            j = i + 1 if i != len(node_list) - 1 else 0
            distance += Solution.distance_ij_accurate(node_list[i], node_list[j])
        return distance

    @staticmethod
    def calc_fitness_of_sd(sd: list, nodes_list):
        """
        计算sd(城市序号数组)的fitness即TSP环路路径质量
        采用路径总距离求倒数
        注：这里为了放大距离间的差别，对距离的倒数求幂进行放大
        """
        nodes_list = [nodes_list[idx] for idx in sd]
        path_distance = Solution.calc_path_distance(nodes_list)
        return math.pow(50000/path_distance, 8), path_distance

    def update_best_fitness(self, new_fitness, new_td, new_sd, new_distance):
        """
        对比新fitness与目前保存的最优fitness
        如果新更优则替换并返回True，否则返回False
        """
        if self.fitness is None:
            self.fitness = new_fitness
            self.td, self.sd = new_td, new_sd
            self.distance = new_distance
            return True
        elif self.fitness <= new_fitness:
            self.fitness = new_fitness
            self.td, self.sd = new_td, new_sd
            self.distance = new_distance
            return True
        else:
            return False

    @staticmethod
    def binary_find_index_in_array(val, array, start, end):
        """二分查找法，从数组中找出值的索引"""
        mid = (start + end)//2
        if val == array[start]:
            return start
        if val == array[end]:
            return end
        if val == array[mid]:
            return mid
        elif val < array[mid]:
            return Solution.binary_find_index_in_array(val, array, start, mid-1)
        else:
            return Solution.binary_find_index_in_array(val, array, mid+1, end)

    @staticmethod
    def find_first_index_of_equal_value_in_array(td: list, index):
        """的数组中index左右找出与index位置值相等的最小索引"""
        if index == 0:
            return index
        while td[index] == td[index - 1]:
            index -= 1
        return index

    @staticmethod
    def find_min_index_in_lable_array(index, sorted_td, labels):
        """
        先找出在index索引左右与sorted_td[index]值相同的最小索引号
        再从该最小索引号开始人labels中查找第一个末标记的索引，返回该索引
        如：index=2, labesl=[False, False, False, False], sorted_td=[0, 1, 1, 1]，则
          1. index=2, sorted_td[2] = 1，则最小索引号为 index = 1
          2. 从index=1开始搜索labels中第一个末标记的索引，即找到index=1
        """
        min_equal_index = Solution.find_first_index_of_equal_value_in_array(sorted_td, index)
        start_index = min_equal_index
        while True:
            if labels[start_index] is True:
                start_index += 1
            else:
                return start_index

    def random_t(self):
        """随机值：0~10"""
        return random.random() * 10


class ABCSystem(object):
    """abc算法"""
    def __init__(self):
        self.nodes_name2pos, self.nodes_list = ABCSystem.read_data_from_file(data_file)
        ABCSystem.node_count = len(self.nodes_name2pos)
        self.city_num = len(self.nodes_list)  # D维度为城市的数量
        # 食物源列表
        self.solutions = []
        # 当前最优fitness
        self.best_fitness = 0
        self.best_distance = 0
        # 当前最好蜜源对应的td sd
        self.best_td = []
        self.best_sd = []

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

    def init_solutions(self):
        """初始化solution，数量与Employed bees相同"""
        for idx in range(NF):
            self.solutions.append(Solution(idx, self.nodes_list))

    def rand_k(self, i):
        """随机产生k值，范围为[0, NF)的整数，且返回的整数不能等于i
        """
        while True:
            k = random.randrange(0, NF)
            if k != i:
                return k

    def calc_solutions_propability(self):
        """
        根据fitness计算蜜源的概率
        """
        total = 0
        for sol in self.solutions:
            total += sol.fitness
        return [sol.fitness/total for sol in self.solutions]

    @staticmethod
    def wheel_select(probabilities: list):
        """轮盘赌法选出一个目标"""
        rand = random.random()
        index = 0
        total_prob = 0
        for prob in probabilities:
            total_prob += prob
            if total_prob > rand:
                break
            index += 1
        return index

    def new_sd_from_old(self, sol):
        """随机替换old sd"""
        return self.copy_and_random_route(sol)

    def copy_and_random_route(self, sol):
        """复制并产生一个新的数组"""
        len_sd = len(sol.sd)
        copy_route = [sd for sd in sol.sd]
        start_idx = random.randint(1, len_sd-1)
        end_idx = random.randint(1, len_sd - 1)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        s = start_idx
        e = end_idx
        while s < e:
            copy_route[s], copy_route[e] = copy_route[e], copy_route[s]
            s += 1
            e -= 1
        return copy_route

    def generate_new_candidate_td(self, sol_i):
        """根据论文公式(2)生成新候选蜜源(candidate solution)"""
        new_td = [None for _ in range(self.city_num)]
        k = self.rand_k(sol_i.index)
        sol_k = self.solutions[k]
        for j in range(self.city_num):
            pha = random.random() * 2 - 1  # -1 ~ 1的随机数
            new_td[j] = sol_i.td[j] + pha * (sol_i.td[j] - sol_k.td[j])
        return new_td

    def employed_bees_search(self):
        """
        雇佣蜂搜索
        每只雇佣蜂对应一个蜜源，并根据论文中公式(2)计算新路径，如果新路径更优则替换当前蜜源
        """
        for i in range(NF):
            sol_i = self.solutions[i]
            sol_i.search_counter += 1
            # new_td = self.generate_new_candidate_td(sol_i)
            # new_sd = Solution.update_sd_from_td(self.city_num, new_td)
            new_td = []
            new_sd = self.new_sd_from_old(sol_i)
            new_fitness, new_distance = Solution.calc_fitness_of_sd(new_sd, self.nodes_list)
            if sol_i.update_best_fitness(new_fitness, new_td, new_sd, new_distance):
                sol_i.search_counter = 0   # 找到更好的解则清0

    def onlooker_bees_search(self):
        """
        跟随蜂按雇佣蜂找到的蜜源的质量，计算概率后根据轮盘赌法选择一个蜜源，并前往蜜源搜索
        跟随蜂同样采用论文中的公式(2)计算新路径，如果找到更优的新路径则替换当前蜜源
        """
        for i in range(NF):
            sol_probabilities = self.calc_solutions_propability()
            select_idx = ABCSystem.wheel_select(sol_probabilities)
            select_sol = self.solutions[select_idx]
            select_sol.search_counter += 1
            # new_td = self.generate_new_candidate_td(select_sol)
            # new_sd = Solution.update_sd_from_td(self.city_num, new_td)
            new_td = []
            new_sd = self.new_sd_from_old(select_sol)
            new_fitness, new_distance = Solution.calc_fitness_of_sd(new_sd, self.nodes_list)
            if select_sol.update_best_fitness(new_fitness, new_td, new_sd, new_distance):
                select_sol.search_counter = 0  # 找到更好的解则清0

    def scout_bees_search(self):
        """侦查蜂随机搜索
        当雇佣蜂指定轮数（食物耗尽）后一直没有找到更优的路径，则变为侦查蜂随机搜索（重新随机初始化蜜源）
        """
        for sol in self.solutions:
            # 超出次数，变为侦查蜂找出一个新的蜜源
            if sol.search_counter != 0 and sol.search_counter >= FOOD_EXHAUSTED_LOOP_NUM:
                self.solutions[sol.index] = Solution(sol.index, self.nodes_list)

    def memorize_best_solution(self):
        """保存当前最好的蜜源信息"""
        for sol in self.solutions:
            if self.best_fitness < sol.fitness:
                self.best_fitness = sol.fitness
                self.best_td = [tid for tid in sol.td]
                self.best_sd = [sid for sid in sol.sd]
                self.best_distance = sol.distance

    def run_once(self, loop_counter):
        """一轮迭代"""
        self.employed_bees_search()
        self.onlooker_bees_search()
        self.scout_bees_search()
        self.memorize_best_solution()
        if loop_counter % FOOD_EXHAUSTED_LOOP_NUM == 0:
            print("loop: %s, fitness: %s, distance: %s" %
                  (loop_counter, self.best_fitness, self.best_distance))
        return self.best_sd

    def run(self):
        """算法入口"""
        # 初始化食物源
        self.init_solutions()
        loop_counter = 0
        while True:
            loop_counter += 1
            self.run_once(loop_counter)
            if loop_counter == MAX_LOOP_NUM:
                break


if __name__ == "__main__":
    abc = ABCSystem()
    abc.run()
    output_png = output_png.format(dist=round(abc.best_distance, 3))
    draw_map(abc.best_sd, output_png)
    print("--------------------------------------------")
    print("best distance: %s, path: %s" % (abc.best_distance, abc.best_sd))
    print("output image: %s" % (output_png, ))
