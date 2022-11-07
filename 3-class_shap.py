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

######################################################################
# print("Train에서 잘못 분류된 결과 값")
# print(np.where(y_train != y_pred)[0])
# incorrect_data = np.array(x_train)[np.where(y_train != y_pred)[0]]
# incorrect_data = pd.DataFrame(incorrect_data, columns=total_columns)
# incorrect_data.to_csv("train_incorrect.csv", index = False)
# print("-----------------------------")
#######################################################################

##### test evaluation #####
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))



#############################
# 데이터 컴증하는데 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
verification_dataset = pd.read_csv(f"./dataset/week_gamble/221025_labeling.csv")
y_verification_dataset = verification_dataset['label']

verification_dataset = transform_dataframe(verification_dataset, x_test.columns)


# a = model.predict_proba(verification_dataset)
# a = np.max(a, axis=1)

# print(np.where(a < 0.90))
# exit()
#########################################################################################


# y_pred = model.predict(verification_dataset)
# accuracy = accuracy_score(y_verification_dataset, y_pred)

# print(accuracy)

# print(np.array(y_verification_dataset))
# print(y_pred)

# print(np.where(y_verification_dataset != y_pred)[0])





######################################################################
# print("Test에서 잘못 분류된 결과 값")
# print(np.where(y_test != y_pred)[0])
# incorrect_data = np.array(x_test)[np.where(y_test != y_pred)[0]]
# incorrect_data = pd.DataFrame(incorrect_data, columns=total_columns)
# incorrect_data.to_csv("test_incorrect.csv", index = False)
# print("-----------------------------")
#######################################################################

feature_names = total_columns

# particular_0 = pd.DataFrame(np.array(x_train)[np.where(y_train==0)], columns= feature_names)
# particular_1 = pd.DataFrame(np.array(x_train)[np.where(y_train==1)], columns= feature_names)
# particular_2 = pd.DataFrame(np.array(x_train)[np.where(y_train==2)], columns= feature_names)

# explainer
explainer = shap.TreeExplainer(model, seed=seed)
shap_values = explainer.shap_values(x_test)

# label_0 = shap_values[1]

# # position = np.argsort(label_0)[-10:]

# # key_0 =  np.array(feature_names)[position]

# # col_0 = []

# # for i in range(len(key_0)):
# #     for j in range(10):
# #         col_0.append(key_0[i][j])

# # from collections import Counter

# # print(dict(Counter(col_0)))
# # # print(sorted(dict(Counter(col_0)).items()))
# # exit()

# position = np.argsort(label_0)[:]

# print(np.array(feature_names)[position][:,74])

# # shap_values = explainer.shap_values([x_test.iloc[0]])
# exit()


# print(x_test.shape)
# print(shap_values[0].shape)
# print(shap_values)

# exit()
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
# print(len(feat_shap_all))

# kk = np.sort(avg_shap)[-300:]

# for i in range(299, 0, -1):
#     i = kk[i]
#     print(feature_names[np.where(i == avg_shap)[0][0]])

# for i in range(299, 0, -1):
#     i = kk[i]
#     print(i)


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
y_pred = model_shap.predict(x_train_shap)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

######################################################################
# print("Shap train에서 잘못 분류한 결과 값")
# print(np.where(y_train != y_pred)[0])
# incorrect_data = np.array(x_train_shap)[np.where(y_train != y_pred)[0]]
# incorrect_data = pd.DataFrame(incorrect_data, columns=shap_columns)
# incorrect_data.to_csv("shap_train_incorrect.csv", index = False)
# print("-----------------------------")
######################################################################

y_pred = model_shap.predict(x_test_shap)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

######################################################################
# print("Shap test에서 잘못 분류한 결과 값")
# print(np.where(y_test != y_pred)[0])
# incorrect_data = np.array(x_test_shap)[np.where(y_test != y_pred)[0]]
# incorrect_data = pd.DataFrame(incorrect_data, columns=shap_columns)
# incorrect_data.to_csv("shap_test_incorrect.csv", index = False)
# print("-----------------------------")
#######################################################################


###################################################################
# 데이터 검증하는데~~~
verification_dataset = transform_dataframe(verification_dataset, x_test.columns)

y_pred = model.predict(verification_dataset)
accuracy = accuracy_score(y_verification_dataset, y_pred)

print(accuracy)

print(np.array(y_verification_dataset))
print(y_pred)

print(np.where(y_verification_dataset != y_pred)[0])









exit()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig('conf_shap.png')

# comparison with model feature importance
feat_importance = list(feature_names[model.feature_importances_ > 0])
print(len(feat_importance))

# filter columns
x_train_im = x_train[feat_importance]
x_test_im = x_test[feat_importance]

##### training #####
try:
    # load model if possible
    model_im = pickle.load(open('3-class-im.pt','rb'))

except:
    model_im = xgb.XGBClassifier(n_estimators=200,
                              max_depth=10,
                              learning_rate=0.5,
                              min_child_weight=0,
                              tree_method='gpu_hist',
                              sampling_method='gradient_based',
                              reg_alpha=0.2,
                              reg_lambda=1.5,
                              random_state=seed)

    st = time.time()
    model_im.fit(x_train_im, y_train)
    ed = time.time()

    # print('[*] time to train importance:', ed-st)

    # pickle.dump(model_im, open('3-class-im.pt','wb'))

##### evaluation #####
y_pred = model_im.predict(x_train_im)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

y_pred = model_im.predict(x_test_im)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# plt.savefig('conf_im.png')

# xgb.plot_importance(model, max_num_features=50)
# plt.savefig('base.png')
# xgb.plot_importance(model_shap, max_num_features=50)
# plt.savefig('shap.png')
# xgb.plot_importance(model_im, max_num_features=50)
# plt.savefig('im.png')

print(len(model.feature_names_in_))
print(len(model_shap.feature_names_in_))
print(len(model_im.feature_names_in_))

print(set(feat_shap_all) == set(feat_importance))