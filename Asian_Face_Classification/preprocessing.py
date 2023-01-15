import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
from tqdm import tqdm           # 진행 상황을 알려줌 tqdm 안에 배열을 넣어서 사용(따로 print  x)
from time import sleep

path = "./origin_data"
file_paths = glob.glob(os.path.join(path, '*', "*", '*.jpg'))

#라벨 정의
labels = ["2030M", "2030F", "4050M", "4050F", "60M", "60F"]

#다시 저장할 곳
rfile_path = "./split_data"

#이미지 나눔
# 파일 경로 예시 ./origin_data\\72\\112\\858972-0.jpg
for img_path in tqdm(file_paths):

    # 데이터 나누기
    # ['./origin_data', '20', '111', '100060-0.jpg']
    split_path = img_path.split("\\")

    age = split_path[1]
    sex = split_path[2]

    # 문자열 숫자로 바꾸기
    age = int(age)

    if age >= 20 and age < 40:

        #남녀 구별
        if sex == "111":

            img = cv2.imread(img_path)

            # 기존 폴더 생성
            directory = rfile_path + "/" + "2030" + "/" + "M"
            if not os.path.exists(directory):
                os.makedirs(directory)

            save_path = directory + "/" + split_path[3]

            # 이미지 다시 저장
            cv2.imwrite(save_path, img)

        elif sex == "112":

            img = cv2.imread(img_path)

            # 기존 폴더 생성
            directory = rfile_path + "/" + "2030" + "/" + "F"
            if not os.path.exists(directory):
                os.makedirs(directory)

            save_path = directory + "/" + split_path[3]

            # 이미지 다시 저장
            cv2.imwrite(save_path, img)

    elif age >= 40 and age < 60:

        if sex == "111":

            img = cv2.imread(img_path)

            # 기존 폴더 생성
            directory = rfile_path + "/" + "4050" + "/" + "M"
            if not os.path.exists(directory):
                os.makedirs(directory)

            save_path = directory + "/" + split_path[3]

            # 이미지 다시 저장
            cv2.imwrite(save_path, img)

        elif sex == "112":

            img = cv2.imread(img_path)

            # 기존 폴더 생성
            directory = rfile_path + "/" + "4050" + "/" + "F"
            if not os.path.exists(directory):
                os.makedirs(directory)

            save_path = directory + "/" + split_path[3]

            # 이미지 다시 저장
            cv2.imwrite(save_path, img)

    elif age > 60:

        if sex == "111":

            img = cv2.imread(img_path)

            # 기존 폴더 생성
            directory = rfile_path + "/" + "60" + "/" + "M"
            if not os.path.exists(directory):
                os.makedirs(directory)

            save_path = directory + "/" + split_path[3]

            # 이미지 다시 저장
            cv2.imwrite(save_path, img)

        elif sex == "112":

            img = cv2.imread(img_path)

            # 기존 폴더 생성
            directory = rfile_path + "/" + "60" + "/" + "F"
            if not os.path.exists(directory):
                os.makedirs(directory)

            save_path = directory + "/" + split_path[3]

            # 이미지 다시 저장
            cv2.imwrite(save_path, img)
