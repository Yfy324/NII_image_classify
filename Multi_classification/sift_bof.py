import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn import preprocessing
from dataset_split import imlist
import pandas as pd
import joblib


def path_label(dir_path, batch_size):
    """
    :param dir_path: 待生成bof特征的图片总路径（可包含不同类别）
    :param k: bof聚类时，类别数量 hyperparameter
    :return: sift_bof_feature + label (0:cat 1:dog)
    """
    label_names = os.listdir(dir_path)

    image_paths = []      # 所有图片对应的路径
    image_classes = []    # 所有图片对应的label
    class_id = 0          # 用于标记类别数，方便编码类别
    for label_name in label_names:
        dir = os.path.join(dir_path, label_name)
        class_path = imlist(dir)
        image_paths += class_path
        image_classes += [class_id] * len(class_path) * batch_size
        class_id += 1
    return image_paths, image_classes


def sift_descriptor_voc(image_paths, k):
    """
    :param image_paths:
    :param k: 词数
    :return:
    """
    # 创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()

    # 特征提取与描述子生成
    des_list = []

    for image_path in image_paths:
        im = cv2.imread(image_path)
        im = cv2.resize(im, (256, 256))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kpts = sift.detect(im)                # find the keypoints
        kpts, des = sift.compute(im, kpts)    # kp为关键点列表，des为numpy的数组，维度是关键点数目×128
        des_list.append((image_path, des))
        print("image file path : ", image_path)

    # 描述子向量
    # 将des_list中的des提出来，按行重新排列整合成新的array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))   # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组

    # 100 聚类 K-Means from: scipy.cluster.vq
    voc, variance = kmeans(descriptors, k, 1)

    return descriptors, voc, des_list


def bof_features(image_paths, voc, des_list, k, batch_size):
    # 生成特征直方图
    # vq: 将图片描述向量(m*n)与视觉字典中的特征进行匹配，返回word(m 与voc中特征对应的索引) 和 distance(与最近质心对应的距离)
    img_features = np.zeros((batch_size*len(image_paths), k), "float32")
    for i in range(batch_size*len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            img_features[i][w] += 1   # 统计匹配某一特征的次数

    nbr_occurrences = np.sum((img_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(img_features) + 1) / (1.0 * nbr_occurrences + 1)), 'float32')

    img_features = img_features * idf
    img_features = preprocessing.normalize(img_features, norm='l2')

    return img_features, idf


def data_merge(image_classes, image_paths, img_features):
    y = np.array(image_classes).reshape((len(image_classes), 1))
    img_data = np.hstack((img_features, y))
    return img_data






