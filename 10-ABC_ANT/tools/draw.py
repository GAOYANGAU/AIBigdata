#############################################
#   将路径数据画到地图上去
#
from PIL import Image, ImageDraw, ImageFont
from tools import Node, Pos


filepath = './resources/wowmap/wow.jpg'
outputimg = './out.png'
posfile = './resources/wowmap/飞行点坐标.txt'
cr = 35  # 圆的半径
font = ImageFont.truetype("./resources/wowmap/simhei.ttf", size=50)
linewidth = 10  # 路径线宽
circle_color = (255, 0, 0, 255)
text_color = (0, 0, 0, 255)


def read_data_from_file(file_path):
    """读取数据返回节点名字到坐标的map与城市列表
    """
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


# 读取城市坐标数据
node_name2pos, node_list = read_data_from_file(posfile)


def draw_map(node_indexes: list, output_img=outputimg):
    """由城市序号在地图上画出路径"""
    # 环路
    if node_indexes[len(node_indexes)-1] != node_indexes[0]:
        node_indexes.append(node_indexes[0])
    # 开始画直线与圆
    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    for i in range(len(node_indexes)):
        j = i + 1 if i != len(node_indexes) - 1 else 0
        index_i, index_j = node_indexes[i], node_indexes[j]
        node_i = node_list[index_i]
        node_j = node_list[index_j]
        draw.line((node_i.pos.log, node_i.pos.lat, node_j.pos.log, node_j.pos.lat),
                  fill=(256, 0, 0, 256), width=linewidth)
    for node in node_list:
        x = node.pos.log
        y = node.pos.lat
        draw.ellipse((x - cr, y - cr, x + cr, y + cr), fill=circle_color)
        if node.index < 10:
            pos = (x - 0.25 * cr, y - 0.7 * cr)
        else:
            pos = (x - 0.65 * cr, y - 0.7 * cr)
        draw.text(pos, str(node.index), font=font, fill=text_color)
    img.save(output_img, 'JPEG')


if __name__ == '__main__':
    indexes = [0, 17, 2, 19, 20, 5, 6, 21, 26, 28, 16, 31, 30, 29, 27, 15, 11, 14, 25, 13, 12, 24, 9, 10, 7, 23, 8, 22, 4, 3, 18, 1]
    # indexes = [node.index for node in node_list]
    draw_map(indexes)
