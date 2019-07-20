"""
使用训练好的模型进行推理测试
Contributor：lujin
Reviewer：xionglongfei
"""

from keras.applications.vgg19 import VGG19
from keras.models import load_model
from PIL import Image
import numpy as np
import os.path


num_classes = 10
model = load_model('retrain.h5') # 加载训练好的模型


while True:
    # 输入图片
    img_path = input('Please input picture file to predict ( input Q to exit ):  ')
    if img_path == 'Q':
        break
    if not os.path.exists(img_path):
        print("file not exist!")
        continue
    img = Image.open(img_path)
	
    # 缩放图片到网络输入尺寸
    ori_w,ori_h = img.size
    new_w = 32
    new_h = 32
    img = img.resize((new_w,new_h))
    x = np.asarray(img, dtype='float32')
    x[:, :, 0] = x[:, :, 0] - 123.680
    x[:, :, 1] = x[:, :, 1] - 116.779
    x[:, :, 2] = x[:, :, 2] - 103.939
    x = np.expand_dims(x, axis=0)

    cifar10_labels = np.array([
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'])
	# 使用预训练过的模型进行预测，并显示预测结果
    results = model.predict(x).flatten().tolist()
    ind = results.index(max(results))
    #print(results)
    print('Predicted:', cifar10_labels[ind])
    pass
