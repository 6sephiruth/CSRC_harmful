from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import xgboost as xgb
import numpy as np
import pandas as pd

from utils import *

import pickle
import time

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

seed = 1

top_list = ['먹튀', '검증', '사이트']

# 485
white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))
white_dataset = white_dataset[top_list]
white_dataset['label'] = 0
print("정상 데이터셋 크기")
print(white_dataset.shape)

# 220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704, raw_gamble_recent, 221025
gamble_dataset = pd.read_csv(f"./dataset/week_gamble/raw_gamble_recent.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
gamble_dataset = gamble_dataset[top_list]
gamble_dataset['label'] = 1
print("도박 데이터셋 크기")
print(gamble_dataset.shape)

## 신규 키워드 넣기 ##
# size_gamble_dataset = len(gamble_dataset)
# gamble_dataset['임규민'] = 0
# gamble_dataset['임규민'][-int(3):] = 333

# 632
ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))
ad_dataset = ad_dataset[top_list]
ad_dataset['label'] = 2
print("광고 데이터셋 크기")
print(ad_dataset.shape)

white_train = white_dataset.sample(frac=0.8, random_state=seed)
white_test = white_dataset.drop(white_train.index)

gamble_train = gamble_dataset.sample(frac=0.8, random_state=seed)
gamble_test = gamble_dataset.drop(gamble_train.index)

ad_train = ad_dataset.sample(frac=0.8, random_state=seed)
ad_test = ad_dataset.drop(ad_train.index)

##### preprocessing #####
init_train = pd.concat([white_train, gamble_train, ad_train])
init_train.fillna(0, inplace=True)

init_test = pd.concat([white_test, gamble_test, ad_test])
init_test.fillna(0, inplace=True)

total_columns = init_train.columns.drop('label')

x_train = init_train.drop('label', axis=1)
y_train = init_train['label']

x_test = init_test.drop('label', axis=1)
y_test = init_test['label']


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dt_clf = DecisionTreeClassifier(random_state = 1004)
dt_clf_model = dt_clf.fit(x_train, y_train)

y_pred = dt_clf_model.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print(accuracy)

y_pred = dt_clf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.tree import export_graphviz

# .dot 파일로 export 해줍니다
export_graphviz(dt_clf_model, out_file='tree.dot')

# 생성된 .dot 파일을 .png로 변환
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=600'])

# jupyter notebook에서 .png 직접 출력
from IPython.display import Image
Image(filename = 'decistion-tree.png')