import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from dataset_split import imlist
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa


def read_img(path):
    im = cv2.imread(path)
    im = cv2.resize(im, (300, 300))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def batch_aug(k, img):
    ia.seed(k)
    imgs = [img, img, img, img, img]  # 改
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25), translate_percent=0.1),
        iaa.Crop(percent=(0, 0.2)),
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
    return des_list[0], des_list[1], des_list[2], des_list[3], des_list[4]  # 改


def sift_bof_features(dir_path, k, batch_size):
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

    # 创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()

    # 特征提取与描述子生成
    des_list = []

    for image_path in image_paths:
        des_2 = img_descriptor(image_path, batch_size)
        for i in range(batch_size):
            des_list.append(des_2[i])
        # des_list.append(des)
        print("image file path : ", image_path)

    # 描述子向量
    # 将des_list中的des提出来，按行重新排列整合成新的array
    descriptors = des_list[0]
    for descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))   # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组

    # 100 聚类 K-Means from: scipy.cluster.vq
    voc, variance = kmeans(descriptors, k, 1)

    # 生成特征直方图
    # vq: 将图片描述向量(m*n)与视觉字典中的特征进行匹配，返回word(m 与voc中特征对应的索引) 和 distance(与最近质心对应的距离)
    im_features = np.zeros((batch_size*len(image_paths), k), "float32")
    for i in range(batch_size*len(image_paths)):
        words, distance = vq(des_list[i], voc)
        for w in words:
            im_features[i][w] += 1   # 统计匹配某一特征的次数

    y = np.array(image_classes).reshape((batch_size*len(image_paths), 1))
    img_data = np.hstack((im_features, y))

    return img_data


# 实现动词词频与出现频率统计
def bof_frequency(im_features):
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(im_features) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
    return nbr_occurences, idf

# # 尺度化 / 标准化: 按均值中心化，以标准差缩放
# stdSlr = StandardScaler().fit(im_features)   # fit：计算均值、标准差，transformer能得到标准化之后的数据
# im_features = stdSlr.transform(im_features)


if __name__ == '__main__':
    train_path = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/train/'
    n_features = 15
    b_size = 5
    data = sift_bof_features(train_path, n_features, b_size)
    data1 = pd.DataFrame(data)
    data1.to_csv('aug5_train.csv')
    # val_path = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/validation/'
    # data_val = sift_bof_features(val_path, n_features, b_size)
    # data2 = pd.DataFrame(data_val)
    # data2.to_csv('aug4_validation.csv')
