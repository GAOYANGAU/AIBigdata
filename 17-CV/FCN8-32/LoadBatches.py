import numpy as np
import cv2
import glob
import itertools
import random


def get_image_arr(im):
    #去均值
    img = im.astype(np.float32)

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    return img


def get_segmentation_arr(seg, n_classes, input_height, input_width):
    """
    生成图片的labels
    :param seg: annotation图
    :param n_classes: 分类默认15
    :param input_height: 图片的高
    :param input_width: 图片的宽
    :return: 生成的标签labels
    """
    seg_labels = np.zeros((input_height, input_width, n_classes))

    for c in range(n_classes):
        seg_labels[:, :, c] = (seg == c).astype(int)

    seg_labels = np.reshape(seg_labels, (-1, n_classes))
    return seg_labels


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width):
    """
    训练图片和对应的标签生成迭代
    :param images_path: 训练图片的路径
    :param segs_path: 标注图片的路径
    :param batch_size: 默认16
    :param n_classes: 默认15
    :param input_height: 默认320
    :param input_width: 默认320
    :return: 输出X和Y对
    """
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.jpg") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        x = []
        y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 0)

            assert im.shape[:2] == seg.shape[:2]

            assert im.shape[0] >= input_height and im.shape[1] >= input_width

            xx = random.randint(0, im.shape[0] - input_height)
            yy = random.randint(0, im.shape[1] - input_width)

            im = im[xx:xx + input_height, yy:yy + input_width]
            seg = seg[xx:xx + input_height, yy:yy + input_width]

            x.append(get_image_arr(im))
            y.append(
                get_segmentation_arr(
                    seg,
                    n_classes,
                    input_height,
                    input_width))

        yield np.array(x), np.array(y)


if __name__ == '__main__':
    G = image_segmentation_generator("data/train_img/",
                                     "data/train_img_ann/",
                                     batch_size=16, n_classes=15, input_height=320, input_width=320)
    x, y = G.__next__()
    print(x.shape, y.shape)
