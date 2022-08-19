from sklearn.metrics import accuracy_score

import tensorflow as tf
import xgboost as xgb
import numpy as np
import pandas as pd

from utils import *

from sklearn.model_selection import cross_val_score

import pickle

seed = 1

# load datasets
# 1233
gamble_dataset = pd.read_csv("./dataset/raw_gamble.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))

# 632
ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))

# 485
white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))


# 데이터셋 나누기 학습용 80% 테스트용 20%
gamble_train = gamble_dataset.sample(frac=0.8, random_state=seed)
gamble_test = gamble_dataset.drop(gamble_train.index)

ad_train = ad_dataset.sample(frac=0.8, random_state=seed)
ad_test = ad_dataset.drop(ad_train.index)

white_train = white_dataset.sample(frac=0.8, random_state=seed)
white_test = white_dataset.drop(white_train.index)

###########################################
# 1단계 모델 만들기 (도박+광고 vs 정상)
white_train['label'] = 0
gamble_train['label'] = 1
ad_train['label'] = 2

white_test['label'] = 0
gamble_test['label'] = 1
ad_test['label'] = 2

model1_train = pd.concat([white_train, gamble_train, ad_train])
model1_train.fillna(0, inplace=True)

total_columns = model1_train.columns

model1_test = pd.concat([white_test, gamble_test, ad_test])
model1_test.fillna(0, inplace=True)

x_train = np.array(model1_train.drop('label', axis=1))
y_train = np.array(model1_train['label'])

x_test = np.array(model1_test.drop('label', axis=1))
y_test = np.array(model1_test['label'])

# define model
model_1 = xgb.XGBClassifier(n_estimators=200,
                            max_depth=10,
                            learning_rate=0.5,
                            min_child_weight=0,
                            tree_method='gpu_hist',
                            sampling_method='gradient_based',
                            reg_alpha=0.2,
                            reg_lambda=1.5)

model_1.fit(x_train, y_train)

print("(도박+광고 vs 정상) model 1 test 정확도")
y_pred = model_1.predict(x_test)
y_pred = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

short_shap_name, short_shap_value = report_shap(x_train, total_columns,  model_1)

print(short_shap_name[:20])
print(short_shap_value[:20])


# x_full = np.concatenate((x_train, x_test), axis = 0)
# y_full = np.concatenate((y_train, y_test), axis = 0)

# cross_week_5 = cross_val_score(model_1, x_full, y_full, cv=5) # model, train, target, cross validation
# cross_week_10 = cross_val_score(model_1, x_full, y_full, cv=10) # model, train, target, cross validation

# print(cross_week_5)
# print(cross_week_10)
