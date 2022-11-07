from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import pickle

seed = 1

# load datasets
gamble_dataset = pd.read_csv("./dataset/raw_gamble.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))

ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))

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
ad_train['label'] = 1

white_test['label'] = 0
gamble_test['label'] = 1
ad_test['label'] = 1

model1_train = pd.concat([gamble_train, ad_train, white_train])
model1_train.fillna(0, inplace=True)

total_columns = model1_train.columns

model1_test = pd.concat([gamble_test, ad_test, white_test])
model1_test.fillna(0, inplace=True)

model1_train = shuffle(model1_train, random_state=seed)
model1_test = shuffle(model1_test, random_state=seed)

x_train = np.array(model1_train.drop('label', axis=1))
y_train = np.array(model1_train['label'])

x_test = np.array(model1_test.drop('label', axis=1))
y_test = np.array(model1_test['label'])

try:
    # load saved model
    model_1 = pickle.load(open('model1.pt','rb'))

except:
    # define model
    model_1 = MLPClassifier()
    model_1.fit(x_train, y_train)
    pickle.dump(model_1, open('model1.pt','wb'))

print("(도박+광고 vs 정상) model 1 test 정확도")
y_pred = model_1.predict(x_test)
y_pred = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

###########################################
# 2단계 모델 만들기 (도박 vs 광고)
gamble_train = pd.DataFrame(gamble_train, columns=total_columns)
gamble_train['label'] = 0
ad_train['label'] = 1

gamble_test = pd.DataFrame(gamble_test, columns=total_columns)
gamble_test['label'] = 0
ad_test['label'] = 1

model2_train = pd.concat([gamble_train, ad_train])
model2_train.fillna(0, inplace=True)

model2_test = pd.concat([gamble_test, ad_test])
model2_test.fillna(0, inplace=True)

model2_train = shuffle(model2_train, random_state=seed)
model2_test = shuffle(model2_test, random_state=seed)

x_train = np.array(model2_train.drop('label', axis=1))
y_train = np.array(model2_train['label'])

x_test = np.array(model2_test.drop('label', axis=1))
y_test = np.array(model2_test['label'])

try:
    # load saved model
    model_2 = pickle.load(open('model2.pt','rb'))

except:
    # define model
    model_2 = MLPClassifier()
    model_2.fit(x_train, y_train)
    pickle.dump(model_2, open('model2.pt','wb'))

print("(도박 vs 광고) model 2 정확도")
y_pred = model_2.predict(x_test)
y_pred = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

###########################################
# 3가지 모델 분류 방법 실험
white_test['label'] = 0
gamble_test['label'] = 1
ad_test['label'] = 2

df_init = pd.concat([gamble_test, ad_test, white_test])
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
print("Accuracy: %.2f" % (accuracy * 100.0))
