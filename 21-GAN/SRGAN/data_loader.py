import os
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        
        fold_path = r'./datasets/'
        path = [os.path.join(fold_path, x) for x in os.listdir(fold_path)]
        
        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = scipy.misc.imresize(img, self.img_res)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # 随机翻转
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)