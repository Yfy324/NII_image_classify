import cv2
import numpy as np
from scipy.cluster.vq import *
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
import joblib
from sift_bof import path_label, data_merge
from sklearn import preprocessing


def read_img(path):
    im = cv2.imread(path)
    im = cv2.resize(im, (256, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def batch_aug(k, img):
    ia.seed(k)
    imgs = []
    for i in range(k):
        imgs.append(img)  # 改
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        # iaa.Affine(rotate=(-25, 25), translate_percent=0.1),
        iaa.Crop(percent=(0, 0.3))
    ])
    images_aug = seq(images=imgs)
    return images_aug


def img_descriptor(path, batch_size):
    image = read_img(path)
    images = batch_aug(batch_size, image)
    des_list = []
    for i in range(len(images)):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(images[i], None)  # find the keypoints
        kp, des = sift.compute(images[i], kp)
        des_list.append(des)
    return des_list  # 改


def generate_descriptors(image_paths):
    # 创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()

    # 特征提取与描述子生成
    des_list = []

    for image_path in image_paths:
        im1 = read_img(image_path)
        kpts = sift.detect(im1)  # find the keypoints
        kpts, des = sift.compute(im1, kpts)  # kp为关键点列表，des为numpy的数组，维度是关键点数目×128
        des_list.append(des)
        # des_2 = img_descriptor(image_path, batch_size)
        # for i in range(batch_size):
        #     des_list.append(des_2[i])
        print("image file path : ", image_path)

    return des_list


def aug_generate_descriptors(image_paths, batch_size):
    # 创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()
    b = batch_size - 1

    # 特征提取与描述子生成
    des_list = []

    for image_path in image_paths:
        im1 = read_img(image_path)
        kpts = sift.detect(im1, None)  # find the keypoints
        kpts, des = sift.compute(im1, kpts)  # kp为关键点列表，des为numpy的数组，维度是关键点数目×128
        des_list.append(des)
        im1s = batch_aug(b, im1)
        for i in range(len(im1s)):
            kpt = sift.detect(im1s[i], None)  # find the keypoints
            kpt, des = sift.compute(im1s[i], kpt)  # kp为关键点列表，des为numpy的数组，维度是关键点数目×128
            des_list.append(des)
        print("image file path : ", image_path)

    return des_list


def generate_features(image_paths, batch_size, des_list, voc, k, idf):
    # 生成特征直方图
    # vq: 将图片描述向量(m*n)与视觉字典中的特征进行匹配，返回word(m 与voc中特征对应的索引) 和 distance(与最近质心对应的距离)
    img_features = np.zeros((batch_size*len(image_paths), k), "float32")
    for i in range(batch_size*len(image_paths)):
        words, distance = vq(des_list[i], voc)
        for w in words:
            img_features[i][w] += 1   # 统计匹配某一特征的次数

    img_features = img_features * idf
    img_features = preprocessing.normalize(img_features, norm='l2')

    return img_features

