"""
predict
Contributor：lujin
Reviewer：xionglongfei
"""
import LoadBatches
from Models import FCN32, FCN8
import glob
import cv2
import numpy as np
import random

n_classes = 11

key = "fcn8"

method = {
    "fcn32": FCN32.fcn32,
    "fcn8": FCN8.fcn8_model,
        }

images_path = "data/val_img/"
segs_path = "data/val_img_ann/"

input_height = 320
input_width = 320

colors = [
    (random.randint(
        0, 255), random.randint(
            0, 255), random.randint(
                0, 255)) for _ in range(n_classes)]

##########################################################################


def label2color(colors, n_classes, seg):
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
    for c in range(n_classes):
        seg_color[:, :, 0] += ((seg == c) *
                               (colors[c][0])).astype('uint8')
        seg_color[:, :, 1] += ((seg == c) *
                               (colors[c][1])).astype('uint8')
        seg_color[:, :, 2] += ((seg == c) *
                               (colors[c][2])).astype('uint8')
    seg_color = seg_color.astype(np.uint8)
    return seg_color


def get_center_offset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy


images = sorted(
    glob.glob(
        images_path +
        "*.jpg") +
    glob.glob(
        images_path +
        "*.png") +
    glob.glob(
        images_path +
        "*.jpeg"))
segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                       glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

m = method[key](11, 320, 320)  # 有自定义层时，不能直接加载模型
m.load_weights("output/%s_model.h5" % key)


for i, (imgName, segName) in enumerate(zip(images, segmentations)):

    print("%d/%d %s" % (i + 1, len(images), imgName))

    im = cv2.imread(imgName, 1)
    xx, yy = get_center_offset(im.shape, input_height, input_width)
    im = im[xx:xx + input_height, yy:yy + input_width, :]

    seg = cv2.imread(segName, 0)
    seg = seg[xx:xx + input_height, yy:yy + input_width]

    pr = m.predict(np.expand_dims(LoadBatches.get_image_arr(im), 0))[0]
    pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)

    cv2.imshow("img", im)
    cv2.imshow("seg_predict_res", label2color(colors, n_classes, pr))
    cv2.imshow("seg", label2color(colors, n_classes, seg))

    cv2.waitKey()
