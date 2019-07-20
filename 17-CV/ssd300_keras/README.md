#使用Keras实现SSD300#
#数据集下载：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
#get_data_from_XML.py 用于标签生成，将标签生成voc2007.pkl文件
#数据集存放位置：
       ./VOC2007/JPEGImages/   中存放训练图片
       VOC2007.pkl  标签文件
运行train.py训练模型
运行test.py 测试模型

#weights文件夹中放入test用到的训练好的模型
#checkpoints文件夹中存放训练时生成的模型
#prior_boxes_ssd300VGG16.pkl 是SSD中的Default box（Prior Box） 
