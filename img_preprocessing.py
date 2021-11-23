# from __future__ import print_function   版本不兼容问题
# from __future__ import absolute_import

import numpy as np 
from skimage import data_dir, io, transform, color
import cv2
import matplotlib.pyplot as plt 

from skimage import data_dir

data_dir = '/home/yfy/Desktop/projects/Git_pj/test/SIFT-BoF/SIFT-BoF-Main/data/pattern/'
# strr = data_dir + '/*.*'
# strr = data_dir + '/*.jpg'
# print(strr)
print(data_dir)

def img_preprocessing(funs, **args):               # 图像预处理 1张
    rgb = io.imread(funs)
    rgb = cv2.resize(rgb, (256,256))
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    # dst = transform.resize(gray, (256*256))
    # return dst
    return rgb

def img_processing(categories='cat'):
    print(data_dir + categories + '/*.*')
    strr = data_dir + categories
    coll = io.ImageCollection(strr+'/*.*', load_func=img_preprocessing)  # 批量处理图像
    for i in range(len(coll)):
        io.imsave(strr+'/gray/'+np.str(i)+'.jpg',coll[i])
    print(coll[1].shape)
    print(len(coll))

    return coll

if __name__ == '__main__':
    img_processing()
    