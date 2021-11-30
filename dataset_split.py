import os
import shutil


def imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def dir_creat(original_dataset_dir, base_dir):
    training_names = os.listdir(original_dataset_dir)

    # The directory where we will store our smaller dataset
    os.mkdir(base_dir)

    dic_dir = os.path.join(base_dir, 'dic')
    os.mkdir(dic_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    for i in range(len(training_names)):
        img_path = img_split(original_dataset_dir, training_names[i])

        r_dic = 80
        r_train = int(0.6 * len(img_path))
        r_val = int(0.8 * len(img_path))
        r_test = len(img_path)
        fnames = ['{}.{}.jpg'.format(training_names[i], m1) for m1 in range(r_test)]

        # Directory with our train_i pictures
        dic_i_dir = os.path.join(dic_dir, training_names[i])
        os.mkdir(dic_i_dir)
        for n0 in range(r_dic):
            src = img_path[n0]
            des = dic_i_dir + '/' + fnames[n0]
            shutil.copy(src, des)

        train_i_dir = os.path.join(train_dir, training_names[i])
        os.mkdir(train_i_dir)
        for n1 in range(r_dic, r_train):
            src = img_path[n1]
            des = train_i_dir + '/' + fnames[n1]
            shutil.copy(src, des)

        # Directory with our validation_i pictures
        validation_i_dir = os.path.join(validation_dir, training_names[i])
        os.mkdir(validation_i_dir)
        for n2 in range(r_train, r_val):
            src = img_path[n2]
            des = validation_i_dir + '/' + fnames[n2]
            shutil.copy(src, des)

        # Directory with our test_i pictures
        test_i_dir = os.path.join(test_dir, training_names[i])
        os.mkdir(test_i_dir)
        for n3 in range(r_val, r_test):
            src = img_path[n3]
            des = test_i_dir + '/' + fnames[n3]
            shutil.copy(src, des)


def img_split(train_path, training_name):
    """
    Get all the path to the images and save them in a list
    image_paths and the corresponding label in image_paths
    """
    image_paths = []      # 所有图片对应的路径
    src = os.path.join(train_path, training_name)
    class_path = imlist(src)
    image_paths += class_path
    return image_paths


if __name__ == '__main__':
    original = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/'
    base = '/home/yfy/Desktop/projects/ImageNet/dogs_vs_cats/cats_and_dogs_small/'
    dir_creat(original, base)
