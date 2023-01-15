import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
from PIL import Image

'''
-split_data
    -train
        -2030
            -M
                -이미지 파일.jpg
            -F
        -4050
        -60
    -val
    -test
'''

# 경로설정 예시: ./split_data/train
# 라벨 값은 str형이면 안된다.

label_dict = {"2030M": 0, "2030F": 1, "4050M": 2, "4050F": 3, "60M": 4, "60F": 5}


class custom_dataset(Dataset):

    def __init__(self, file_paths, transform=None):
        # 파일 경로 설정
        self.images_path = glob.glob(os.path.join(file_paths, "*", '*', '*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        # 이미지 추출
        img_path = self.images_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 라벨 추출
        path_split = img_path.split("\\")
        label = path_split[1] + path_split[2]
        label = int(label_dict[label])
        # 라벨 텐서 변환
        label = torch.tensor(label)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.images_path)

# 테스트
# test = custom_dataset('./split_data/train')
# print(test[0])
