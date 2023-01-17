# 금속 표면 결함 분류 프로젝트

MS AI School 프로젝트에서 제공 받은 금속 표면 결함 이미지를 파이토치를 사용하여 분류하는 문제를 해결한다.

## 학습 목표
json파일을 파이썬으로 읽고 bbox와 label을 추출한 후 총 10가지의 결함으로 분류하는 학습 모델을 생성한다.

## 코드 진행 순서

<ol>
    <li>preprocessing을 통하여 json 파일의 bbox를 통해 resize한 후 label을 추출하여 분류한다.</li>
    <li>Data_Split을 통하여 train, val 그리고 test 3가지로 이미지를 분류한다.</li>
    <li>main의 train val()을  동작하여 model 생성 및 acc 기록 파일 생성</li>
    <li>main의 model_eval()을 동작하여 model을 평가한다.</li>
</ol>

## 이미지 폴더 구조

![img](./md_img/detail.png)