
class Pos(object):
    """经纬度或像素坐标"""

    def __init__(self, log, lat):
        self.log = float(log)
        self.lat = float(lat)


class Node(object):
    """城市节点"""

    def __init__(self, node_index, name, pos: Pos):
        self.name = name  # 节点名称
        self.index = node_index  # 节点编号
        self.pos = pos  # 节点坐标
