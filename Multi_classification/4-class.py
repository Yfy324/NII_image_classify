import dataset_split
from img_fea_generator import *
from sift_bof import *

if __name__ == '__main__':
    # original = '/home/yfy/Desktop/projects/ImageNet/fly/'
    # base = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/'
    # dataset_split.dir_creat(original, base)
    #
    # dic_path = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/dic/'
    # n_features = 150
    # im_paths_dic, im_classes_dic = path_label(dic_path, 1)
    # describe_dic, dic, d_list_dic = sift_descriptor_voc(im_paths_dic, n_features)
    # im_features_dic, im_distribution = bof_features(im_paths_dic, dic, d_list_dic, n_features, 1)
    # im_data = data_merge(im_classes_dic, im_paths_dic, im_features_dic)
    # joblib.dump((im_distribution, n_features, dic), "4c_bof150_features.pkl", compress=3)

    train_path = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/train'
    im_distribution, n_features, dic = joblib.load('4c_bof150_features.pkl')
    aug = 2
    aug_size = 1 + aug
    im_paths_tr, im_classes_tr = path_label(train_path, aug_size)
    d_list_tr = aug_generate_descriptors(im_paths_tr, aug_size)
    for i in range(len(d_list_tr)):
        if d_list_tr[i] is None:
            d_list_tr[i] = d_list_tr[i-1]
            print("None appears", i)
    im_features_tr = generate_features(im_paths_tr, aug_size, d_list_tr, dic, n_features, im_distribution)
    data_train = data_merge(im_classes_tr, im_paths_tr, im_features_tr)
    train1 = pd.DataFrame(data_train)
    train1.to_csv('4c_2_train_150.csv')

    # val_path = '/home/yfy/Desktop/projects/ImageNet/fly/fly_small/validation/'
    # im_distribution, n_features, dic = joblib.load('4c_bof150_features.pkl')
    # b_size = 1
    # im_paths_val, im_classes_val = path_label(val_path, b_size)
    # d_list_val = generate_descriptors(im_paths_val)
    # im_features_val = generate_features(im_paths_val, b_size, d_list_val, dic, n_features, im_distribution)
    # data_val = data_merge(im_classes_val, im_paths_val, im_features_val)
    # val1 = pd.DataFrame(data_val)
    # val1.to_csv('4c_1_val_150.csv')
