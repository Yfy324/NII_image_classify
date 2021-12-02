import cv2
import numpy as np
from dataset_split import imlist
import os
import joblib

# 定义路径
dic_path = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/dic/'
tra_path = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/train/'
val_path = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/validation/'


label_names = os.listdir(dic_path)
paths_dic0 = imlist(os.path.join(dic_path, label_names[0]))
paths_dic1 = imlist(os.path.join(dic_path, label_names[1]))

paths_tra0 = imlist(os.path.join(tra_path, label_names[0]))
paths_tra1 = imlist(os.path.join(tra_path, label_names[1]))

paths_val0 = imlist(os.path.join(val_path, label_names[0]))
paths_val1 = imlist(os.path.join(val_path, label_names[1]))


# 创建两个sift实例，一个提取关键点，一个提取描述符
detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

# 创建flann的匹配器实例
flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

# 创建BOW训练器
bow_kmeans_trainer = cv2.BOWKMeansTrainer(100)

# 为bow训练器制定的簇数为40，接下来初始化bow提取器（bow_extractor），视觉词汇作为bow的输入，在测试图像中会检测这些视觉词汇
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)


# 每个类读入8个图像的sift特征，sift函数返回图像的描述符：
def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


for i in range(len(paths_dic0)):
    bow_kmeans_trainer.add(extract_sift(paths_dic0[i]))
    bow_kmeans_trainer.add(extract_sift(paths_dic1[i]))

# 运用训练器的cluster函数来创建视觉词汇，降为bow_extractor指定返回的词汇，以便他能够从测试图像中提取描述符
voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)


# 定义一个函数，该函数返回基于bow的描述符提取器计算得到的描述符，并用数组来存储20个样本的描述符和label
def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))


train_data, train_labels = [], []
for m in range(len(paths_tra0)):
    train_data.extend(bow_features(paths_tra0[m]))
    train_labels.append(0)
    train_data.extend(bow_features(paths_tra1[m]))
    train_labels.append(1)

val_data, val_labels = [], []
for n in range(len(paths_val0)):
    val_data.extend(bow_features(paths_val0[n]))
    val_labels.append(0)
    val_data.extend(bow_features(paths_val1[n]))
    val_labels.append(1)

joblib.dump((train_data, train_labels, val_data, val_labels), 'new_bof_xy.pkl', compress=3)


# 创建一个svm实例，并对其进行训练
def svm_pred():
    svm = cv2.ml.SVM_create()
    svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_labels))

    # 定义predict函数，并返回预测
    def predict(fn):
        f = bow_features(fn)
        p = svm.predict(f)
        print(fn, "\t", p[1][0][0])
        return p

    # 最后读取要检测的图像，并通过predict进行判断
    car = "1.jpg"
    car_img = cv2.imread(car)
    car_predict = predict(car)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if car_predict[1][0][0] == 0.0:
        cv2.putText(car_img, 'Cat Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        cv2.putText(car_img, 'Cat Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('BOW + SVM Success', car_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
