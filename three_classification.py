import tensorflow as tf
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

import pandas as pd

gamble_dataset = pd.read_csv("./dataset/raw_gamble.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))

ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))

white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))

# dataset shuffle
gamble_dataset = gamble_dataset.sample(frac=1)
ad_dataset = ad_dataset.sample(frac=1)
white_dataset = white_dataset.sample(frac=1)

# 데이터셋 나누기 학습용 80% 테스트용 20%
gamble_train = gamble_dataset.sample(frac=0.8)
gamble_test = gamble_dataset.drop(gamble_train.index)

ad_train = ad_dataset.sample(frac=0.8)
ad_test = ad_dataset.drop(ad_train.index)

white_train = white_dataset.sample(frac=0.8)
white_test = white_dataset.drop(white_train.index)

###########################################
# 1단계 모델 만들기 (도박 vs 정상)
white_train['label'] = 0
gamble_train['label'] = 1
ad_train['label'] = 1

df_init = pd.concat([gamble_train, ad_train, white_train])
df_init.fillna(0, inplace=True)

total_columns = df_init.columns

init_x = np.array(df_init.drop('label', axis=1))
init_y = np.array(df_init['label'])

# define model
model_1 = xgb.XGBClassifier(n_estimators=200,
                            max_depth=10,
                            learning_rate=0.5,
                            min_child_weight=0,
                            tree_method='gpu_hist',
                            sampling_method='gradient_based',
                            reg_alpha=0.2,
                            reg_lambda=1.5)

model_1.fit(init_x, init_y)

print("(도박 vs 정상) model 1 정확도")
y_pred = model_1.predict(init_x)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(init_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

###########################################
# 2단계 모델 만들기 (도박 vs 광고)
gamble_train = pd.DataFrame(gamble_train, columns=total_columns)
gamble_train['label'] = 0
ad_train['label'] = 1

df_init = pd.concat([gamble_train, ad_train])
df_init.fillna(0, inplace=True)

init_x = np.array(df_init.drop('label', axis=1))
init_y = np.array(df_init['label'])

# define model
model_2 = xgb.XGBClassifier(n_estimators=200,
                            max_depth=10,
                            learning_rate=0.5,
                            min_child_weight=0,
                            tree_method='gpu_hist',
                            sampling_method='gradient_based',
                            reg_alpha=0.2,
                            reg_lambda=1.5)

model_2.fit(init_x, init_y)

print("(도박 vs 광고) model 2 정확도")
y_pred = model_2.predict(init_x)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(init_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

###########################################
# 3가지 모델 분류 방법 실험
white_test['label'] = 0
gamble_test['label'] = 1
ad_test['label'] = 2

df_init = pd.concat([white_test, gamble_test, ad_test])
df_init.fillna(0, inplace=True)

init_x = np.array(df_init.drop('label', axis=1))
init_y = np.array(df_init['label'])

# 정상 vs 도박 분류
model1_pred = np.array(model_1.predict(init_x))

print(len(np.where(model1_pred > 0)[0]))
print(init_x.shape)
print(init_x[np.where(model1_pred > 0)].shape)
model2_pred = np.array(model_2.predict(init_x[np.where(model1_pred > 0)]))

prediction_result = np.empty_like(init_y)

prediction_result[np.where(model2_pred > 0)] = 2
prediction_result[np.where(model2_pred < 1)] = 1
prediction_result[np.where(model1_pred < 1)] = 0

accuracy = accuracy_score(prediction_result, init_y)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
