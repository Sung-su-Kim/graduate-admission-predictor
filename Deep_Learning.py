# 대학원 합격 확률 계산 프로그램

# 모델의 성능 향상 방법
# 1. 완벽한 데이터 전처리 -> 성능 향상
# 2. 하이퍼파라미터(모델) 튜닝
# 딥러닝 성능 향상은 실험적으로 연구해야한다


# 데이터 준비 단계

import pandas as pd   # <-- pandas는 파이썬으로 엑셀처럼 데이터 다루고 싶을 때(행과 열) 사용하는 라이브러리 

dataset = pd.read_csv("Dataset.csv")   # <-- csv 파일을 읽겠다는 뜻

# 데이터 전처리 (Data Preprocessing)
dataset.isnull().sum()   #<-- 빈 칸 개수 파악
dataset = dataset.dropna()   # <-- 빈 칸 혹은 NaN 행을 자동 제거
# dataset.fillna(100)   <-- 빈 칸을 원하는 값으로 채움


# 라벨링
label = dataset["admit"].values   # <-- values : 리스트로 관리


featuer = []

for i, rows in dataset.iterrows():  # <-- 데이터프레임을 한 행씩 출력
    featuer.append([rows["gre"], rows["gpa"], rows["rank"]])


# 모델 생성 단계

# 모델 학습에서 파이썬 리스트는 사용 못하고 numpy array라는 것을 사용하기 떄문에 필요 (다차원 리스트, 행렬만들때 사용하는 라이브러리)
import numpy as np

import tensorflow as tf

# keras >> tensorflow 안에 있는 도구 (딥러닝 모델 제작이 간편해짐)
model = tf.keras.models.Sequential([  # <-- 딥러닝 모델 (Sequential 쓰면 신경망 레이어를 쉽게 만들어줌)
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(128, activation = "relu"),   # <-- 히든 레이어 (노드 개수 *노드 개수는 실험적으로 결과 잘 나올 때까지 바꿔줘야 한다)
    tf.keras.layers.Dense(1, activation = "sigmoid"),   # <-- 반환값 한 개 (*딥러닝에서 정수 예측은 힘듦) 0~1의 확률 반환이므로 시그모이드 함수
])


# optimizer : 경사하강법 러닝레이트에서 뺄 값을 상황에 맞게 조정해주는 역할 (Adam을 많이 사용)
# loss : 손실함수 (*확률 문제 혹은 분류 문제에서는 binary_crossentropy를 사용)
# metrics : 모델을 어떤 요소로 평가할 건지 선택
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# 학습 단계 (가중치값 최적화)
model.fit(np.array(featuer), np.array(label), epochs = 100)


# 예측 단계
predict = model.predict(np.array([[750, 3.70, 3], [400, 2.2, 1]]))   # <-- 새로운 데이터로 예측

print(np.round(predict * 100))