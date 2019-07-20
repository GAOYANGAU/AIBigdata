## 代码说明：
该工程代码分为两部分：
1.训练代码：基于vgg19预训练模型，增加了cifar-10分类的层，训练出来具备10分类能力的网络。保存模型为：retrain.h5
2.预测代码：加载训练好的模型，进行预测，可输入一个图片，网络输出最可能的类别。


## 实验环境
python >=3.5
tensorflow >=1.4.1
keras >=2.1.5

## 数据准备
keras提供cifar-10的数据获取以及处理（在代码中已有，不用做修改）
如果网速太慢，可以预先从https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz下载tar.gz包解压后放到~/keras/dataset 文件夹中,目录命名为cifar-10-batches-py 
脚本会自动跳过cifar-10的数据下载过程。

## vgg19预训练模型准备
由于外服速度较慢，读者可自行选择vgg19预训练模型获取获取方式
是否使用拉取到本地的vgg19预训练h5模型，代码差异：
1.从服务器下载时：
filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
filepath = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'   权重路径
#filepath = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
2.从本地读取h5预训练模型时：
#filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
#filepath = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'   权重路径
filepath = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'

从https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5下载vgg19预训练h5模型并放在工程目录



## 预测代码
model_prediction.py

预测时输入测试图片的路径，返回预测的结果。