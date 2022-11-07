#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import time
import os
import pickle

import numpy as np
import xgboost as xgb

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

seed = 1

gamble_dataset = pd.read_csv("./dataset/raw_gamble.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))

ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))

white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))

size_gamble_dataset = len(gamble_dataset)
size_ad_dataset = len(ad_dataset)
size_white_dataset = len(white_dataset)

print(size_gamble_dataset)
print(size_ad_dataset)
print(size_white_dataset)

total_dataset = pd.concat([gamble_dataset, ad_dataset, white_dataset])
total_dataset = total_dataset.fillna(0)

columns_total_data = total_dataset.columns

progress_total_dataset = np.where(total_dataset>0, 1, 0)

pre_total_dataset = pd.DataFrame(progress_total_dataset, columns=columns_total_data)

pre_gamble_dataset = pd.DataFrame(pre_total_dataset.iloc[:1233], columns=columns_total_data)
pre_ad_dataset = pd.DataFrame(pre_total_dataset.iloc[1233:1865], columns=columns_total_data)
pre_white_dataset = pd.DataFrame(pre_total_dataset.iloc[1865:], columns=columns_total_data)

pre_gamble_dataset['correct_label'] = 0
pre_ad_dataset['correct_label'] = 1
pre_white_dataset['correct_label'] = 2

### 데이터셋 train, validation, test 분할

# Shuffle
gamble = pre_gamble_dataset.sample(frac=1, random_state=seed)
ad = pre_ad_dataset.sample(frac=1, random_state=seed)
white = pre_white_dataset.sample(frac=1, random_state=seed)

# 데이터셋 나누기 학습용 90% 테스트용 10%
gamble_train = gamble.sample(frac=0.8, random_state=seed)
gamble_test = gamble.drop(gamble_train.index)

ad_train = ad.sample(frac=0.8, random_state=seed)
ad_test = ad.drop(ad_train.index)

white_train = white.sample(frac=0.8, random_state=seed)
white_test = white.drop(white_train.index)

gamble_validation = gamble_test.sample(frac=0.5, random_state=seed)
gamble_test = gamble_test.drop(gamble_validation.index)

ad_validation = ad_test.sample(frac=0.5, random_state=seed)
ad_test = ad_test.drop(ad_validation.index)

white_validation = white_test.sample(frac=0.5, random_state=seed)
white_test = white_test.drop(white_validation.index)

### 1. gamble vs. ad ###
train = pd.concat([gamble_train, ad_train], axis=0)
validation = pd.concat([gamble_validation, ad_validation], axis=0)
test = pd.concat([gamble_test, ad_test], axis=0)

train = train.sample(frac=1, random_state=seed)
validation = validation.sample(frac=1, random_state=seed)
test = test.sample(frac=1, random_state=seed)

y_train = train['correct_label'].to_numpy() > 0
x_train = pd.DataFrame(train.drop(['correct_label'], axis=1))
x_train = np.array(x_train)

y_val = validation['correct_label'].to_numpy() > 0
x_val = pd.DataFrame(validation.drop(['correct_label'], axis=1))
x_val = np.array(x_val)

y_test = test['correct_label'].to_numpy() > 0
x_test = pd.DataFrame((test.drop(['correct_label'], axis=1)))
x_test = np.array(x_test)

try:
    gam_ad_model = pickle.load(open('gam_ad_model.pt','rb'))

except:
    # train an XGBoost model
    #gam_ad_model = xgb.XGBClassifier(n_estimators=200,
    #                          max_depth=10,
    #                          learning_rate=0.5,
    #                          min_child_weight=0,
    #                          reg_alpha=0.2,
    #                          reg_lambda=1.5)
    gam_ad_model = MLPClassifier()
    gam_ad_model.fit(X=x_train, y=y_train)
    pickle.dump(gam_ad_model, open('gam_ad_model.pt','wb'))

print('[*] Model 1: gamble vs. ad')

y_pred = gam_ad_model.predict(x_val)
accuracy = accuracy_score(y_pred, y_val)
print('Validation Acc.:', accuracy)

y_pred = gam_ad_model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('Test Acc.:', accuracy)

### 2. gamble vs. normal ###
train = pd.concat([gamble_train, white_train], axis=0)
validation = pd.concat([gamble_validation, white_validation], axis=0)
test = pd.concat([gamble_test, white_test], axis=0)

train = train.sample(frac=1, random_state=seed)
validation = validation.sample(frac=1, random_state=seed)
test = test.sample(frac=1, random_state=seed)

y_train = train['correct_label'].to_numpy() > 0
x_train = pd.DataFrame(train.drop(['correct_label'], axis=1))
x_train = np.array(x_train)

y_val = validation['correct_label'].to_numpy() > 0
x_val = pd.DataFrame(validation.drop(['correct_label'], axis=1))
x_val = np.array(x_val)

y_test = test['correct_label'].to_numpy() > 0
x_test = pd.DataFrame((test.drop(['correct_label'], axis=1)))
x_test = np.array(x_test)

try:
    gam_white_model = pickle.load(open('gam_white_model.pt','rb'))

except:
    # train an XGBoost model
    #gam_white_model = xgb.XGBClassifier(n_estimators=200,
    #                          max_depth=10,
    #                          learning_rate=0.5,
    #                          min_child_weight=0,
    #                          reg_alpha=0.2,
    #                          reg_lambda=1.5)
    gam_white_model = MLPClassifier()
    gam_white_model.fit(X=x_train, y=y_train)
    pickle.dump(gam_white_model, open('gam_white_model.pt','wb'))

print('[*] Model 2: gamble vs. normal')

y_pred = gam_white_model.predict(x_val)
accuracy = accuracy_score(y_pred, y_val)
print('Validation Acc.:', accuracy)

y_pred = gam_white_model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('Test Acc.:', accuracy)

### 3. ad vs. normal ###
train = pd.concat([ad_train, white_train], axis=0)
validation = pd.concat([ad_validation, white_validation], axis=0)
test = pd.concat([ad_test, white_test], axis=0)

train = train.sample(frac=1, random_state=seed)
validation = validation.sample(frac=1, random_state=seed)
test = test.sample(frac=1, random_state=seed)

y_train = train['correct_label'].to_numpy() > 1
x_train = pd.DataFrame(train.drop(['correct_label'], axis=1))
x_train = np.array(x_train)

y_val = validation['correct_label'].to_numpy() > 1
x_val = pd.DataFrame(validation.drop(['correct_label'], axis=1))
x_val = np.array(x_val)

y_test = test['correct_label'].to_numpy() > 1
x_test = pd.DataFrame((test.drop(['correct_label'], axis=1)))
x_test = np.array(x_test)

try:
    ad_white_model = pickle.load(open('ad_white_model.pt','rb'))

except:
    # train an XGBoost model
    #ad_white_model = xgb.XGBClassifier(n_estimators=200,
    #                          max_depth=10,
    #                          learning_rate=0.5,
    #                          min_child_weight=0,
    #                          reg_alpha=0.2,
    #                          reg_lambda=1.5)
    ad_white_model = MLPClassifier()
    ad_white_model.fit(X=x_train, y=y_train)
    pickle.dump(ad_white_model, open('ad_white_model.pt','wb'))

print('[*] Model 3: ad vs. normal')

y_pred = ad_white_model.predict(x_val)
accuracy = accuracy_score(y_pred, y_val)
print('Validation Acc.:', accuracy)

y_pred = ad_white_model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('Test Acc.:', accuracy)