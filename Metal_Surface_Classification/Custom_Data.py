import torch
from torch.utils.data import Dataset
import os
import glob
import cv2

label_dict = {"crease": 0, "crescent_gap": 1, "inclusion": 2, "oil_spot": 3, "punching_hole": 4,
              "rolled_pit": 5, "silk_spot": 6, "waist_folding": 7, "water_spot": 8, "welding_line": 9}

class custom_dataset(Dataset):

    def __init__(self, file_paths, transform=None):
        # 파일 경로 설정
        self.images_path = glob.glob(os.path.join(file_paths, '*', '*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        # 이미지 추출
        img_path = self.images_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 라벨 추출
        path_split = img_path.split("\\")
        label = path_split[1]
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