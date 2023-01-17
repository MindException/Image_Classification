import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
from tqdm import tqdm
import json


# 이미지 파일 읽기
path = "./origin_data"
file_paths = glob.glob(os.path.join(path, 'images', '*.jpg'))
# ./origin_data\\images\\img_08_4406743300_00699.jpg


# json 파일 읽기
file_path = "./origin_data/anno/annotation.json"

# 기존 json 파일 읽어오기
with open(file_path, 'r') as file:
    json_data = json.load(file)

# 라벨 읽기
# json_data["img_01_3402617700_00001.jpg"]["anno"][0]["label"]

r_save_path = "./split_data"

if not os.path.exists(r_save_path):
    os.makedirs(r_save_path)

for img_path in tqdm(file_paths):

    # 사진 이름 검출
    img_name = img_path.split("\\")[-1]

    # 이미지 읽기
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 재조정
    detection_place = json_data[img_name]["anno"][0]["bbox"]
    # 예시 [1738, 806, 1948, 993]

    x, y, dx, dy = detection_place
    detect_img = img[y:dy+1, x:dx+1]

    # 저장 경로 설정
    directory = r_save_path + "/" + json_data[img_name]["anno"][0]["label"]

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = directory + "/" + img_name

    cv2.imwrite(save_path, detect_img)