import dataset_split
import features_generator
import img_fea_generator


if __name__ == '__main__':
    original = '/home/yfy/Desktop/projects/ImageNet/fly/'
    base = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/'
    dataset_split.dir_creat(original, base)
