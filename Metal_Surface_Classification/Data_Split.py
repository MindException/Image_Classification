import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

'''
-split_data
    -Product Image
        -train
            - 종류
                - 이미지
        -val
        -test
'''

main_path = "./split_data"


def move_file(dir_name, save_paths, save_labels):
    dir_path = main_path + "/" + dir_name  # train, val, test

    for now_path, save_label in zip(tqdm(save_paths), save_labels):

        # ['./split_data', 'crescent_gap', 'img_01_425007500_01452.jpg']
        file_name = now_path.split("\\")[2]
        label_path = now_path.split("\\")[1]

        # \를 전부 /로 다시 변환하여 path 잡아주기
        now_path = main_path + "/" + label_path
        save_path = dir_path + "/" + label_path

        # 파일 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        now_path = now_path + "/" + file_name
        save_path = save_path + "/" + file_name

        # 모든 파일 옮기기
        shutil.move(now_path, save_path)


file_paths = glob.glob(os.path.join(main_path, "*", '*.jpg'))

label_list = []
for path in file_paths:
    label_data = path.split('\\')[1]
    label_list.append(label_data)

path_train, path_temp, label_train, label_temp = train_test_split(file_paths, label_list, train_size=0.8,
                                                                  stratify=label_list, random_state=2023)

path_val, path_test, label_val, label_test = train_test_split(path_temp, label_temp, train_size=0.5,
                                                              stratify=label_temp, random_state=2023)

# 파일 이동
move_file("train", path_train, label_train)
move_file("val", path_val, label_val)
move_file("test", path_test, label_test)
