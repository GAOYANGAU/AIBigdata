"""
使用SSD训练好的网络进行目标检测
Contributor：lujin
"""

import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from ssd300VGG16 import SSD
from ssd_utils import BBoxUtility
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
np.set_printoptions(suppress=True)
#限制GPU内存
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# set_session(tf.Session(config=config))

if __name__ == '__main__':
    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    NUM_CLASSES = len(voc_classes) + 1

    input_shape = (300, 300, 3)
    model = SSD(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('./weights/VGG16SSD300weights_voc_2007.hdf5', by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)

    inputs = []
    images = []
    img_path = './pics/000002.jpg'
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))

    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    a = model.predict(inputs, batch_size=1)
    b = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # 选择置信度高于0.6
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        plt.show()
