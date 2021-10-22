import os
import random
import shutil
from tqdm import tqdm
if __name__ == '__main__':
    #讲path替换成自己的
    raw_img_path = r'C:\Users\XXX\PycharmProjects\White-box-Cartoonization-master\test_code\test_images'
    cartoonized_img_path = r'C:\Users\XXX\PycharmProjects\White-box-Cartoonization-master\test_code\cartoonized_pic'
    trainA_save_path = r'C:\Users\XXX\PycharmProjects\White-box-Cartoonization-master\test_code\dataset\trainA'
    trainB_save_path = r'C:\Users\XXX\PycharmProjects\White-box-Cartoonization-master\test_code\dataset\trainB'
    testA_save_path = r'C:\Users\XXX\PycharmProjects\White-box-Cartoonization-master\test_code\dataset\testA'
    testB_save_path = r'C:\Users\XXX\PycharmProjects\White-box-Cartoonization-master\test_code\dataset\testB'

    filelist = os.listdir(raw_img_path)

    cartoon_filelist = os.listdir(cartoonized_img_path)

    #从数据集中随机挑选128张test
    seed = [i for i in range(0, len(filelist), 1)]
    list_test = random.sample(seed, 128)


    #制作testA testB
    for i in tqdm(list_test):

        raw_pathA = os.path.join(raw_img_path,filelist[i])
        cartoon_pathA = os.path.join(cartoonized_img_path, cartoon_filelist[i])

        # print(raw_pathA,testA_save_path)
        # print(cartoon_pathA,testB_save_path)

        shutil.move(raw_pathA,testA_save_path)
        shutil.move(cartoon_pathA, testB_save_path)


    # 制作trianA trainB
    new_filelist = os.listdir(raw_img_path)
    new_cartoon_filelist = os.listdir(cartoonized_img_path)
    print('len:', len(new_filelist))

    for i in tqdm(range(len(new_filelist))):
        raw_pathA = os.path.join(raw_img_path,new_filelist[i])
        cartoon_pathA = os.path.join(cartoonized_img_path, new_cartoon_filelist[i])
        #
        # print(raw_pathA,trainA_save_path)
        # print(cartoon_pathA,trainB_save_path)

        shutil.move(raw_pathA,trainA_save_path)
        shutil.move(cartoon_pathA, trainB_save_path)



