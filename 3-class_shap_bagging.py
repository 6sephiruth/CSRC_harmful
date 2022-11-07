from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import xgboost as xgb
import numpy as np
import pandas as pd

from utils import *

import pickle
import time

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

seed = 1

# 485
white_dataset = pd.read_csv("./dataset/raw_white.csv")
white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))
white_dataset['label'] = 0

# 220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704, raw_gamble_recent

gamble_dataset = pd.read_csv(f"./dataset/week_gamble/raw_gamble_recent.csv")
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
ad_test = ad_dataset.drop(ad_train.index)

##### preprocessing #####
init_train = pd.concat([white_train, gamble_train, ad_train])
init_train.fillna(0, inplace=True)

init_test = pd.concat([white_test, gamble_test, ad_test])
init_test.fillna(0, inplace=True)

total_columns = init_train.columns

x_train = init_train.drop('label', axis=1)
y_train = init_train['label']

x_test = init_test.drop('label', axis=1)
y_test = init_test['label']

from sklearn.ensemble import HistGradientBoostingClassifier
try:
    # load model if possible
    model = pickle.load(open('3-class_bagging.pt','rb'))

except:

    model = HistGradientBoostingClassifier(min_samples_leaf=1).fit(x_train, y_train)
    start_time = time.time()
    model.fit(x_train, y_train)

    print("time :", time.time() - start_time)

    #pickle.dump(model, open('3-class_bagging.pt','wb'))

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)


accuracy = accuracy_score(y_train, y_pred_train)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

accuracy = accuracy_score(y_test, y_pred_test)
print("Test accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

# feature_names
feature_names = total_columns.drop('label').values

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
print(len(feat_shap_all))

kk = np.sort(avg_shap)[-20:]

for i in kk:
    print(feature_names[np.where(i == avg_shap)[0][0]])
    print(i)
    print("--------------")

exit()
# filter columns
x_train_shap = x_train[feat_shap_all]
x_test_shap = x_test[feat_shap_all]

try:
    # load model if possible
    model_shap = pickle.load(open('3-class-shap_gnv.pt','rb'))

except:
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    start_time = time.time()
    model.fit(x_train_shap, y_train)

    print("time :", time.time() - start_time)

    pickle.dump(model, open('3-class_svm.pt','wb'))

y_pred_train = model.predict(x_train_shap)
y_pred_test = model.predict(x_test_shap)


accuracy = accuracy_score(y_train, y_pred_train)
print("Train accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")

accuracy = accuracy_score(y_test, y_pred_test)
print("Test accuracy: %.2f" % (accuracy * 100.0))
print("-----------------------------")


exit()


    # pickle.dump(model, open('3-class.pt','wb'))

























##### training #####
try:
    # load model if possible
    model = pickle.load(open('3-class.pt','rb'))

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

    # pickle.dump(model, open('3-class.pt','wb'))

##### evaluation #####
y_pred = model.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig('conf_base.png')

# feature_names
feature_names = total_columns.drop('label').values

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
print(len(feat_shap_all))

kk = np.sort(avg_shap)[-20:]

for i in kk:
    print(feature_names[np.where(i == avg_shap)[0][0]])
    print(i)
    print("--------------")
exit()


# filter columns
x_train_shap = x_train[feat_shap_all]
x_test_shap = x_test[feat_shap_all]

# print(x_train_shap)
# print(x_test_shap)

##### training #####
try:
    # load model if possible
    model_shap = pickle.load(open('3-class-shap.pt','rb'))

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

    # pickle.dump(model_shap, open('3-class-shap.pt','wb'))

##### evaluation #####
y_pred = model_shap.predict(x_train_shap)
accuracy = accuracy_score(y_train, y_pred)
print("Train accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

y_pred = model_shap.predict(x_test_shap)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: %.2f" % (accuracy * 100.0))

print("-----------------------------")

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