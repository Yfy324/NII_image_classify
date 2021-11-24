import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.cluster.vq import *

def cv_img_sift():
    img = cv2.imread('/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/train/cat/cat.2.jpg')
    # gray = cv2.imread('/home/yfy/Downloads/1.jpg', cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = sift.detect(gray, None)                     # find the keypoints
    kp, des = sift.compute(gray, kp)                  # kp为关键点列表，des为numpy的数组，为关键点数目×128
    # kp, des = sift.detectAndCompute(gray,None)

    img = cv2.drawKeypoints(gray, kp, img)            # draw the keypoints
    cv2.imshow('sp', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_img(path):
    im = cv2.imread(path)
    im = cv2.resize(im, (500, 300))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def batch_aug(k, img):
    ia.seed(k)
    imgs = [img, img, img, img]
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25), translate_percent=0.05),
        iaa.Crop(percent=(0, 0.2)),
    ])
    images_aug = seq(images=imgs)
    return images_aug


def img_descriptor(path, n_batch):
    image = read_img(path)
    images = batch_aug(n_batch, image)
    des_list = []
    for i in range(len(images)):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(images[i], None)  # find the keypoints
        kp, des = sift.compute(images[i], kp)
        des_list.append(des)
    return des_list[0], des_list[1], des_list[2], des_list[3]
    # return des_list[0]


if __name__ == '__main__':
    des_l = img_descriptor('1.jpg', 4)
    des_list = []
    for i in range(4):
        des_list.append(des_l[i])

    des_2 = img_descriptor('1.jpg', 4)
    for i in range(4):
        des_list.append(des_2[i])
