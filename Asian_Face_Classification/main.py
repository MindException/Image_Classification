import sys
import os
import glob
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from Custom_Dataset import custom_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd

import warnings

warnings.filterwarnings(action='ignore')

# cuda 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 알부 사용
train_transform = A.Compose([
    A.Resize(width=87, height=87),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(width=87, height=87),
    A.Normalize(),
    ToTensorV2()
])

train_dataset = custom_dataset('./split_data/train', transform=train_transform)
val_dataset = custom_dataset('./split_data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

net = models.shufflenet_v2_x2_0(pretrained=True)
net.fc = nn.Linear(in_features=2048, out_features=6)
net.to(device)

# 에포크 설정
num_epochs = 10

# 손실함수와 옵티마이저 추가
criterion = LabelSmoothingCrossEntropy()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.005)


def train_val():
    best_val_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    os.makedirs('.\\models', exist_ok=True)
    save_path = '.\\models\\best.pt'
    dfForAccuracy = pd.DataFrame(index=list(range(num_epochs)), columns=['Epoch', 'train_loss', 'train_acc',
                                                                         'val_loss', 'val_acc'])

    # 다시 시작하면 폴더 꼭 지워주기
    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv('./ModelAccuracy.csv')['val_acc'].tolist())

    for epoch in range(num_epochs):

        running_loss = 0
        val_losses = 0
        val_acc = 0
        train_acc = 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')

        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = f'Train Epoch [{epoch + 1}/{num_epochs}], loss >> {loss.data:.3f}'

        net.eval()
        with torch.no_grad():
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='green')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_losses += loss.item()

                val_acc += (torch.argmax(outputs, 1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'train_loss'] = round(running_loss / train_steps, 3)
        dfForAccuracy.loc[epoch, 'train_acc'] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, 'val_loss'] = round(val_losses / val_steps, 3)
        dfForAccuracy.loc[epoch, 'val_acc'] = round(val_accuracy, 3)

        print(f'Epoch [{epoch + 1}/{num_epochs}]',
              f'Train Loss : {(running_loss / train_steps):.3f}',
              f'Train Acc : {train_accuracy:.3f},'
              f'Val Loss : {(val_losses / val_steps):.3f}',
              f' Val Acc : {val_accuracy:.3f}')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch == num_epochs - 1:
            torch.save(net.state_dict(), '.\\models\\last.pt')

        if epoch == num_epochs - 1:
            dfForAccuracy.to_csv('.\\ModelAccuracy.csv', index=False)


def model_eval(model):

    model.eval()
    total = 0
    correct = 0

    test_dataset = custom_dataset('./split_data/test', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for i, (image, labels) in enumerate(tqdm(test_loader)):
            image, label = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)

            total += image.size(0)
            correct += (label == argmax).sum().item()

        acc = correct / total * 100
        print("acc for {} image : {:.2f}%".format(
            total, acc
        ))


if __name__ == '__main__':

    # 학습 모델
    # train_val()

    # 테스트 모델
    net.load_state_dict(torch.load("./models/best.pt", map_location=device))
    model_eval(net)
