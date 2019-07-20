# 算法说明
## 说明
这里提供了三个优化算法：人工蜂群算法、蚁群算法、退火模拟算法，并将之用于解决TSP问题，这里的TSP问题求的是在游戏WOW魔兽世界诺森德大陆地图上，在中立阵营与部落阵营的飞行点间（假设飞行点间可相互直接线飞行），寻找一条穿过所有飞行点的环路，且每个飞行点只能过一次，要求该环路的总长度尽可能短。

* 执行环境：`python3.6`及以上
* 地图推荐使用魔兽世界飞行点坐标图，见`./resources/wowmap/wow_pos.jpg`，图中标出的序号对应`./resources/wowmap/飞行点坐标.txt`中的行号
* `./resources/wowmap/wow.jpg`图片用于生成计算后的路径图，不可删除
* 计算生成的路径结果图片保存在文件夹：`./outputs`中，命名格式：`{算法名}_{环路长度}.jpg`

## 人工蜂群算法
参考自文献：`Pathak N, Tiwari S P. Travelling salesman problem using bee colony with SPV[J]. organization, 2012, 13: 18.`

执行算法：
```py
python abc.py
```
结果见`outputs/abc_{}.jpg`

## 蚁群算法
参考自文献：`Colorni A, Dorigo M, Maniezzo V. Distributed optimization by ant colonies[C]//Proceedings of the first European conference on artificial life. 1991, 142: 134-142`

执行算法：
```py
python ant.py
```
结果见`outputs/ant_{}.jpg`

## 退火模拟算法
```py
python sa.py
```
结果见`outputs/sa_{}.jpg`
