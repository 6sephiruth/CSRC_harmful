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

# 485
white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))
white_dataset['label'] = 0
print("정상 데이터셋 크기")
print(white_dataset.shape)

# 220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704, raw_gamble_recent, 221025
gamble_dataset = pd.read_csv(f"./dataset/week_gamble/raw_gamble_recent.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
gamble_dataset['label'] = 1
print("도박 데이터셋 크기")
print(gamble_dataset.shape)

# 632
ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))
ad_dataset['label'] = 2
print("광고 데이터셋 크기")
print(ad_dataset.shape)


##### preprocessing #####
init_train = pd.concat([white_dataset, gamble_dataset, ad_dataset])
init_train.fillna(0, inplace=True)

total_columns = init_train.columns.drop('label')

x_train = init_train.drop('label', axis=1)
y_train = init_train['label']


##### training #####
try:
    # load model if possible
    model = pickle.load(open('./model/3-class_full_train.pt','rb'))

except:
    model = xgb.XGBClassifier(n_estimators=200,
                              max_depth=10,
                              learning_rate=0.5,
                              min_child_weight=0,
                              tree_method='gpu_hist',
                              sampling_method='gradient_based',
                              reg_alpha=0.2,
                              reg_lambda=1.5,
                              random_state=seed)
    st = time.time()
    model.fit(x_train, y_train)
    ed = time.time()

    # print('[*] time to train baseline:', ed-st)

    pickle.dump(model, open('./model/3-class_full_train.pt','wb'))

##### train evaluation #####
y_pred = model.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

feature_names = total_columns

# explainer
explainer = shap.TreeExplainer(model, seed=seed)
shap_values = explainer.shap_values(x_train)

# filtering mode
#FILTER = "by-order"         # 각 클래스별 top 100 키워드 추출
FILTER = "by-thresh"        # 각 클래스별 SHAP이 0보다 큰 키워드 추출

# placeholder for feature sets
feat_shap = []

n_class = 3
for cls in range(n_class):
    attr = shap_values[cls]

    # calculate mean(|SHAP values|) for each class
    avg_shap = np.abs(attr).mean(0)
    l = len(avg_shap)

    # filtering by ordering
    if FILTER == 'by-order':
        idxs = np.argpartition(avg_shap, l-100)[-100:]
        keywords = set(feature_names[idxs])

    # filtering by thresholding
    elif FILTER == 'by-thresh':
        keywords = set(feature_names[avg_shap > 0])

    feat_shap.append(keywords)

# keywords from shap
from functools import reduce
feat_shap_all = list(reduce(set.union, feat_shap))

# filter columns
x_train_shap = x_train[feat_shap_all]

shap_columns = x_train_shap.columns

##### training #####
try:
    # load model if possible
    model_shap = pickle.load(open('./model/3-class-shap_full_train.pt','rb'))

except:
    model_shap = xgb.XGBClassifier(n_estimators=200,
                              max_depth=10,
                              learning_rate=0.5,
                              min_child_weight=0,
                              tree_method='gpu_hist',
                              sampling_method='gradient_based',
                              reg_alpha=0.2,
                              reg_lambda=1.5,
                              random_state=seed)

    st = time.time()
    model_shap.fit(x_train_shap, y_train)
    ed = time.time()

    # print('[*] time to train shap:', ed-st)

    # pickle.dump(model_shap, open('./model/3-class-shap.pt','wb'))

##### evaluation #####
y_pred = model_shap.predict(x_train_shap)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

print(accuracy)
print()
exit()

new_dataset = pd.read_csv(f"./dataset/week_gamble/221025_labeling.csv")
y_new_dataset = new_dataset['label']

new_dataset = transform_dataframe(new_dataset, shap_columns)

pred = model_shap.predict(new_dataset)

print(np.where(pred != y_new_dataset))


exit()