import cv2
import numpy as np
from dataset_split import imlist
import os
import joblib
import imgaug as ia
from imgaug import augmenters as iaa

# 定义路径
dic_path = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/dic/'
tra_path = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/train/'
val_path = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/validation/'


label_names = os.listdir(dic_path)
paths_dic0 = imlist(os.path.join(dic_path, label_names[0]))
paths_dic1 = imlist(os.path.join(dic_path, label_names[1]))
paths_dic2 = imlist(os.path.join(dic_path, label_names[2]))
paths_dic3 = imlist(os.path.join(dic_path, label_names[3]))

paths_tra0 = imlist(os.path.join(tra_path, label_names[0]))
paths_tra1 = imlist(os.path.join(tra_path, label_names[1]))
paths_tra2 = imlist(os.path.join(tra_path, label_names[2]))
paths_tra3 = imlist(os.path.join(tra_path, label_names[3]))

paths_val0 = imlist(os.path.join(val_path, label_names[0]))
paths_val1 = imlist(os.path.join(val_path, label_names[1]))
paths_val2 = imlist(os.path.join(val_path, label_names[2]))
paths_val3 = imlist(os.path.join(val_path, label_names[3]))


# 创建两个sift实例，一个提取关键点，一个提取描述符
detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

# 创建flann的匹配器实例
flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

# 创建BOW训练器
bow_kmeans_trainer = cv2.BOWKMeansTrainer(120)

# 为bow训练器制定的簇数为40，接下来初始化bow提取器（bow_extractor），视觉词汇作为bow的输入，在测试图像中会检测这些视觉词汇
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)


# 每个类读入8个图像的sift特征，sift函数返回图像的描述符：
def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


for i in range(len(paths_dic0)):
    bow_kmeans_trainer.add(extract_sift(paths_dic0[i]))
    bow_kmeans_trainer.add(extract_sift(paths_dic1[i]))
    bow_kmeans_trainer.add(extract_sift(paths_dic2[i]))
    bow_kmeans_trainer.add(extract_sift(paths_dic3[i]))

# 运用训练器的cluster函数来创建视觉词汇，降为bow_extractor指定返回的词汇，以便他能够从测试图像中提取描述符
voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)


# 定义一个函数，该函数返回基于bow的描述符提取器计算得到的描述符，并用数组来存储20个样本的描述符和label
def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))


def batch_aug(k, path):
    img = cv2.imread(path, 0)
    ia.seed(1)
    imgs = []
    for i1 in range(k):
        imgs.append(img)  # 改
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-40, 30)),
        # iaa.Affine(rotate=(-25, 25), translate_percent=0.1),
        iaa.Crop(percent=(0, 0.3))
    ])
    images_aug = seq(images=imgs)
    return images_aug


def aug_bow_features(img):
    return extract_bow.compute(img, detect.detect(img))


n = 3
train_data, train_labels = [], []
for m0 in range(len(paths_tra0)):
    train_data.extend(bow_features(paths_tra0[m0]))
    train_labels.append(0)
    ims = batch_aug(n, paths_tra0[m0])
    for k0 in range(n):
        train_data.extend(aug_bow_features(ims[k0]))
        train_labels.append(0)

for m1 in range(len(paths_tra1)):
    train_data.extend(bow_features(paths_tra1[m1]))
    train_labels.append(1)
    ims = batch_aug(n, paths_tra1[m1])
    for k1 in range(n):
        train_data.extend(aug_bow_features(ims[k1]))
        train_labels.append(1)

for m2 in range(len(paths_tra2)):
    train_data.extend(bow_features(paths_tra2[m2]))
    train_labels.append(2)
    ims = batch_aug(n, paths_tra2[m2])
    for k2 in range(n):
        if aug_bow_features(ims[k2]) is None:
            train_data.extend(train_data[len(train_data)-1])
            train_labels.append(2)
        else:
            train_data.extend(aug_bow_features(ims[k2]))
            train_labels.append(2)

for m3 in range(len(paths_tra1)):
    train_data.extend(bow_features(paths_tra3[m3]))
    train_labels.append(3)
    ims = batch_aug(n, paths_tra3[m3])
    for k3 in range(n):
        if aug_bow_features(ims[k3]) is None:
            train_data.extend(train_data[len(train_data)-1])
            train_labels.append(3)
        else:
            train_data.extend(aug_bow_features(ims[k3]))
            train_labels.append(3)


# val_data, val_labels = [], []
# for n in range(len(paths_val0)):
#     val_data.extend(bow_features(paths_val0[n]))
#     val_labels.append(0)
#     val_data.extend(bow_features(paths_val1[n]))
#     val_labels.append(1)
#     val_data.extend(bow_features(paths_val1[n]))
#     val_labels.append(2)
#     val_data.extend(bow_features(paths_val1[n]))
#     val_labels.append(3)


# joblib.dump((train_data, train_labels, val_data, val_labels), '4c_new_bof_xy1.pkl', compress=3)


