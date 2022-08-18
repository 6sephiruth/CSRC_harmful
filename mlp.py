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
white_train['label'] = 0
gamble_train['label'] = 1
ad_train['label'] = 2

white_test['label'] = 0
gamble_test['label'] = 1
ad_test['label'] = 2

xy_train = pd.concat([gamble_train, ad_train, white_train])
xy_train.fillna(0, inplace=True)

total_columns = xy_train.columns

xy_test = pd.concat([gamble_test, ad_test, white_test])
xy_test.fillna(0, inplace=True)

xy_train = shuffle(xy_train, random_state=seed)
xy_test = shuffle(xy_test, random_state=seed)

x_train = np.array(xy_train.drop('label', axis=1))
y_train = np.array(xy_train['label'])

x_test = np.array(xy_test.drop('label', axis=1))
y_test = np.array(xy_test['label'])

try:
    # load saved model
    mlp = pickle.load(open('mlp.pt','rb'))
    print(mlp.classes_)

except:
    # define model
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    pickle.dump(mlp, open('mlp.pt','wb'))

print("mlp train 정확도")
y_pred = mlp.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

print("mlp test 정확도")
y_pred = mlp.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))