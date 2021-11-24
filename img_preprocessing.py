import numpy as np
from skimage import data_dir, io, transform, color
import cv2
import matplotlib.pyplot as plt
from skimage import data_dir
import imageio
import imgaug as ia
from imgaug import augmenters as iaa


data_dir = '/home/yfy/Desktop/projects/Git_pj/test/SIFT-BoF/SIFT-BoF-Main/data/dc/'
# strr = data_dir + '/*.*'
# strr = data_dir + '/*.jpg'
# print(strr)
print(data_dir)


def batch_aug(k, img):
    ia.seed(k)
    imgs = [img, img, img, img]
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25), translate_percent=0.05),
        iaa.Crop(percent=(0, 0.2)),
    ])
    images_aug = seq(images=imgs)
    return images_aug


def img_preprocessing(funs, **args):               # 图像预处理 1张
    rgb = io.imread(funs)
    rgb = cv2.resize(rgb, (300, 300))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ims = batch_aug(4, gray)
    return ims[0], ims[1], ims[2], ims[3]


def img_processing(categories='dog'):
    print(data_dir + categories + '/*.*')
    strr = data_dir + categories
    coll = io.ImageCollection(strr+'/*.*', load_func=img_preprocessing)  # 批量处理图像
    for i in range(len(coll)):
        io.imsave(strr+'/gray/'+np.str(i)+'.jpg', coll[i])
    print(coll[1].shape)
    print(len(coll))
    return coll


if __name__ == '__main__':
    a = img_processing()
    