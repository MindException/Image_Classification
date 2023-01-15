import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil  # 고수준 파일 연산

'''
-split_data
    -Product Image
        -train
            - 나이(2030, 4050, 60)
                - 성별(M/F)
                    -   images
        -val
        -test
'''

# X_train, X_test, Y_train, Y_test
# X는 데이터 경로이며, Y는 레이블이다.

main_path = "./split_data"

#각자의 파일 저장
def move_file(dir_name, save_paths, save_labels):

    dir_path = main_path + "/" + dir_name

    #./split_data\2030\M\584789-0.jpg
    #['2030', 'M']
    for now_path, save_label in zip(tqdm(save_paths), save_labels):

        file_name = now_path.split("\\")[-1]

        #\를 전부 /로 다시 변환하여 path 잡아주기
        now_path = main_path + "/" + save_label[0] + "/" + save_label[1]
        save_path = dir_path + "/" + save_label[0] + "/" + save_label[1]

        # 파일 없으면 생성
        if not os.path.exists(save_path):
            print(dir_path, " 파일 생성")
            os.makedirs(save_path)

        now_path = now_path + "/" + file_name
        save_path = save_path + "/" + file_name

        #모든 파일 옮기기
        shutil.move(now_path, save_path)

file_paths = glob.glob(os.path.join(main_path, '*', "*", '*.jpg'))

label_list = []
for path in file_paths:
    # 나눠질 모양
    # ['./split_data', '60', 'M', '859750-0.jpg']

    # 라벨에는 나이와 성별이 들어가 있다.
    label_data = path.split('\\')
    age = label_data[1]
    sex = label_data[2]
    temp_list = [age, sex]
    label_list.append(temp_list)

# train과 temp 8:2로 나누고 temp를 val과 test로 1:1로 나눈다.
# 분류 문제는 stratify 사용 시 좀 더 좋다고 함
path_train, path_temp, label_train, label_temp = train_test_split(file_paths, label_list, train_size=0.8,
                                                                  stratify=label_list, random_state=2023)

path_val, path_test, label_val, label_test = train_test_split(path_temp, label_temp, train_size=0.5,
                                                              stratify=label_temp, random_state=2023)

#파일 이동
move_file("train" , path_train, label_train)
move_file("val" , path_val, label_val)
move_file("test" , path_test, label_test)