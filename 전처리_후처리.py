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

##### training #####
try:
    # load model if possible
    model = pickle.load(open('./model/3-class.pt','rb'))

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

    pickle.dump(model, open('./model/3-class.pt','wb'))

##### train evaluation #####
y_pred = model.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

##### test evaluation #####
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

feature_names = total_columns

# explainer
explainer = shap.TreeExplainer(model, seed=seed)
shap_values = explainer.shap_values(x_test)

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
x_test_shap = x_test[feat_shap_all]

shap_columns = x_train_shap.columns

##### training #####
try:
    # load model if possible
    model_shap = pickle.load(open('./model/3-class-shap.pt','rb'))

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
y_pred_train = model_shap.predict(x_train_shap)
accuracy = accuracy_score(y_train, y_pred_train)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

y_pred_test = model_shap.predict(x_test_shap)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

############################################################################################
# # 오탐난거 27  58  63  73  74
new_dataset = pd.read_csv(f"./dataset/week_gamble/221025_labeling.csv")
x_new_dataset = transform_dataframe(new_dataset, shap_columns)
y_new_dataset = new_dataset['label']


print(np.where(y_new_dataset != model_shap.predict(x_new_dataset)))



particular_data = x_new_dataset.loc[[58, 63,73, 74]]

particular_data['사이트'] = [10,10,10,10]

print(model_shap.predict(particular_data))

# print(np.sum(particular_data, axis=1))

exit()
explainer = shap.TreeExplainer(model_shap, seed=seed)
shap_values = explainer.shap_values(particular_data)

#print(np.array(x_train_shap)

print(np.array(shap_columns)[np.argsort(shap_values[2], axis=1)][:,-5:])



exit()
# particular_0 = pd.DataFrame(np.array(x_train_shap)[np.where(y_train==0)], columns= shap_columns)
# particular_1 = pd.DataFrame(np.array(x_train_shap)[np.where(y_train==1)], columns= shap_columns)
# particular_2 = pd.DataFrame(np.array(x_train_shap)[np.where(y_train==2)], columns= shap_columns)




particular_data['정보'] = [6,6,6]

print(np.where(y_new_dataset != model_shap.predict(x_new_dataset)))
print(model_shap.predict(particular_data))

exit()


print(x_new_dataset['정보'][58])
x_new_dataset.loc[58, '정보'] = 6
print(x_new_dataset['정보'][58])


print(x_test_shap.shape)
print(x_new_dataset.loc[58].shape)
# print(model_shap.predict(x_new_dataset.loc[58]))
#print(model_shap.predict(x_new_dataset.iloc[58]))

exit()

# explainer = shap.TreeExplainer(model_shap, seed=seed)
# shap_values = explainer.shap_values(x_new_dataset)

# # 오탐난거 27  58  63  73  74

# pred = model_shap.predict(x_new_dataset)
# print(np.where(y_new_dataset != pred))

# # print(np.array(shap_columns)[np.argsort(shap_values[1][27])][-5:])
# # print(np.array(shap_columns)[np.argsort(shap_values[1][32])][-5:])
# # print(np.array(shap_columns)[np.argsort(shap_values[1][58])][-5:])
# # print(np.array(shap_columns)[np.argsort(shap_values[1][63])][-5:])
# # print(np.array(shap_columns)[np.argsort(shap_values[1][73])][-5:])
# # print(np.array(shap_columns)[np.argsort(shap_values[1][74])][-5:])


# print(x_new_dataset['소개'][74])
# print(x_new_dataset['가입'][74])
# print(x_new_dataset['정보'][74])
# print(x_new_dataset['사이트'][74])
# print(x_new_dataset['코드'][74])

exit()

# explainer
explainer = shap.TreeExplainer(model_shap, seed=seed)
shap_values = explainer.shap_values(particular_1)

sort_colums = []

for i in range(len(shap_values[1])):
    for j in range(5):
        sort_colums.append(np.array(shap_columns)[np.argsort(shap_values[1])[:,-5:]][i,j])

from collections import Counter

counter = Counter(sort_colums)
a = dict(counter)

res = sorted(a.items(), key=lambda x:x[1], reverse = True)
print(res)
