# 딥러닝 안쓰고 임계치 값 찾기
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
white_dataset = white_dataset[np.sum(white_dataset, axis=1) != 0]
white_dataset['label'] = 0
print("정상 데이터셋 크기")
print(white_dataset.shape)

# 220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704, raw_gamble_recent
gamble_dataset = pd.read_csv(f"./dataset/week_gamble/raw_gamble_recent.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
gamble_dataset = gamble_dataset[np.sum(gamble_dataset, axis=1) != 0]
gamble_dataset['label'] = 1
print("도박 데이터셋 크기")
print(gamble_dataset.shape)

# 632
ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1))
ad_dataset = ad_dataset[np.sum(ad_dataset, axis=1) != 0]
ad_dataset['label'] = 2
print("광고 데이터셋 크기")
print(ad_dataset.shape)

print()
print()

print(np.mean(white_dataset['사이트']))
print(np.mean(gamble_dataset['사이트']))
print(np.mean(ad_dataset['사이트']))
print("-----------------------")
print(np.std(white_dataset['사이트']))
print(np.std(gamble_dataset['사이트']))
print(np.std(ad_dataset['사이트']))


exit()

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

# white_total_mean = int(np.mean(np.sum(pd.DataFrame(white_train.drop('label', axis=1)), axis=1)))
# gamble_total_mean = int(np.mean(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)), axis=1)))
# ad_total_mean = int(np.mean(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)), axis=1)))

# total_mean = [white_total_mean, gamble_total_mean, ad_total_mean]


# white_total_std = int(np.std(np.sum(pd.DataFrame(white_train.drop('label', axis=1)), axis=1)))
# gamble_total_std = int(np.std(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)), axis=1)))
# ad_total_std = int(np.std(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)), axis=1)))

# total_std = [white_total_std, gamble_total_std, ad_total_std]

# predict_mean = []
# predict_std = []

# for i in np.sum(x_test, axis=1):
#     predict_mean.append(findNearNum(total_mean, i)[0])
#     predict_std.append(findNearNum(total_std, i)[0])


# # print(accuracy_score(predict_mean, np.array(y_test)))
# # print(accuracy_score(predict_std, np.array(y_test)))


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

# feature_names
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

shap_10_list = []

for i in range(10, 0, -1):
    i = np.sort(avg_shap)[-11:][i]
    # print(feature_names[np.where(i == avg_shap)[0][0]])
    shap_10_list.append(feature_names[np.where(i == avg_shap)[0][0]])




exit()

white_total_mean = int(np.mean(np.sum(pd.DataFrame(white_train.drop('label', axis=1)[shap_10_list]), axis=1)))
gamble_total_mean = int(np.mean(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)[shap_10_list]), axis=1)))
ad_total_mean = int(np.mean(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)[shap_10_list]), axis=1)))

total_mean = [white_total_mean, gamble_total_mean, ad_total_mean]


white_total_std = int(np.std(np.sum(pd.DataFrame(white_train.drop('label', axis=1)[shap_10_list]), axis=1)))
gamble_total_std = int(np.std(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)[shap_10_list]), axis=1)))
ad_total_std = int(np.std(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)[shap_10_list]), axis=1)))

total_std = [white_total_std, gamble_total_std, ad_total_std]

predict_mean = []
predict_std = []

for i in np.sum(x_test[feat_shap_all], axis=1):
    predict_mean.append(findNearNum(total_mean, i)[0])
    predict_std.append(findNearNum(total_std, i)[0])

print(accuracy_score(predict_mean, np.array(y_test)))
print(accuracy_score(predict_std, np.array(y_test)))




















# print(int(np.min(np.sum(pd.DataFrame(white_train.drop('label', axis=1)), axis=1))))
# print(int(np.max(np.sum(pd.DataFrame(white_train.drop('label', axis=1)), axis=1))))
# print(int(np.mean(np.sum(pd.DataFrame(white_train.drop('label', axis=1)), axis=1))))
# print(int(np.std(np.sum(pd.DataFrame(white_train.drop('label', axis=1)), axis=1))))
# print("---------------------------------------------------------------")
# print(int(np.min(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)), axis=1))))
# print(int(np.max(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)), axis=1))))
# print(int(np.mean(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)), axis=1))))
# print(int(np.std(np.sum(pd.DataFrame(gamble_train.drop('label', axis=1)), axis=1))))
# print("---------------------------------------------------------------")
# print(int(np.min(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)), axis=1))))
# print(int(np.max(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)), axis=1))))
# print(int(np.mean(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)), axis=1))))
# print(int(np.std(np.sum(pd.DataFrame(ad_train.drop('label', axis=1)), axis=1))))



# exit()
# # 새로운 레이블 생성한다음에 평균
# print(np.mean(np.sum(pd.DataFrame(white_dataset.drop('label', axis=1)), axis=1)))
# print(np.mean(np.sum(pd.DataFrame(gamble_dataset.drop('label', axis=1)), axis=1)))
# print(np.mean(np.sum(pd.DataFrame(ad_dataset.drop('label', axis=1)), axis=1)))
# print("---------------------------------------------------------------")
# print(np.std(np.sum(pd.DataFrame(white_dataset.drop('label', axis=1)), axis=1)))
# print(np.std(np.sum(pd.DataFrame(gamble_dataset.drop('label', axis=1)), axis=1)))
# print(np.std(np.sum(pd.DataFrame(ad_dataset.drop('label', axis=1)), axis=1)))
# print("---------------------------------------------------------------")
# print(np.var(np.sum(pd.DataFrame(white_dataset.drop('label', axis=1)), axis=1)))
# print(np.var(np.sum(pd.DataFrame(gamble_dataset.drop('label', axis=1)), axis=1)))
# print(np.var(np.sum(pd.DataFrame(ad_dataset.drop('label', axis=1)), axis=1)))


# exit()
# count = 4
# print(np.max(white_dataset[shap_list[count]]))
# print(np.mean(white_dataset[shap_list[count]]))
# print(np.std(white_dataset[shap_list[count]]))
# print("----------------------------------------")
# print(np.max(gamble_dataset[shap_list[count]]))
# print(np.mean(gamble_dataset[shap_list[count]]))
# print(np.std(gamble_dataset[shap_list[count]]))
# print("----------------------------------------")
# print(np.max(ad_dataset[shap_list[count]]))
# print(np.mean(ad_dataset[shap_list[count]]))
# print(np.std(ad_dataset[shap_list[count]]))





