from sklearn.metrics import accuracy_score

import tensorflow as tf
import xgboost as xgb
import numpy as np
import pandas as pd

from utils import *

from sklearn.model_selection import cross_val_score

import pickle

seed = 1

# 485
white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))
white_dataset['label'] = 0

gamble_dataset = pd.read_csv(f"./dataset/week_gamble/220502.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
gamble_dataset['label'] = 1

# 632
ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))
ad_dataset['label'] = 2


white_train = white_dataset.sample(frac=0.8, random_state=seed)
white_test = white_dataset.drop(white_train.index)

gamble_train = gamble_dataset.sample(frac=0.8, random_state=seed)
gamble_test = gamble_dataset.drop(gamble_train.index)

ad_train = ad_dataset.sample(frac=0.8, random_state=seed)
ad_test = ad_dataset.drop(ad_dataset.index)


init_train = pd.concat([white_train, gamble_train, ad_train])
init_train.fillna(0, inplace=True)

init_test = pd.concat([white_test, gamble_test, ad_test])
init_test.fillna(0, inplace=True)

total_columns = init_train.columns

x_train = np.array(init_train.drop('label', axis=1))
y_train = np.array(init_train['label'])

x_test = np.array(init_test.drop('label', axis=1))
y_test = np.array(init_test['label'])

model = xgb.XGBClassifier(n_estimators=200,
                            max_depth=10,
                            learning_rate=0.5,
                            min_child_weight=0,
                            tree_method='gpu_hist',
                            sampling_method='gradient_based',
                            reg_alpha=0.2,
                            reg_lambda=1.5)

model = model.fit(x_train, y_train)

y_pred = model.predict(x_train)
y_pred = [round(value) for value in y_pred]
accuracy = accuracy_score(y_train, y_pred)
print("train accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

y_pred = model.predict(x_test)
y_pred = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")


short_shap_name, short_shap_value = report_shap(x_test, total_columns,  model)

print(short_shap_name)
print(short_shap_value)

