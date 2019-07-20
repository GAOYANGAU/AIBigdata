## 实验环境
python3.6
keras 2.2.4  
tensorflow-gpu 1.10.0

数据格式：
	数据都存放在data文件下中
	里面包含示例图片
data/
--train_img 存放训练图片
--train_img_ann 存放训练集标注好的标签图片（一共15类）
--val_img  测试集
--val_img_ann 测试集标注图片
数据集的下载地址：
https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing
或者链接：https://pan.baidu.com/s/1PjEilQpr0dEGLCdpG1GiAw 提取码：cg9g 

训练过程
train.py
设置训练文件的路径，模型预训练权重VGG16。
训练好的模型存放在output文件夹
预测过程
predict.py
确保output文件夹中有训练好的权重

