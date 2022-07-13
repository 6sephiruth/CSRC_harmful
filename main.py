import os
import schedule
import time
import pandas as pd
from datetime import datetime
import tensorflow as tf
import shutil

from utils import *

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

TF_ENABLE_ONEDNN_OPTS = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

TF_ENABLE_ONEDNN_OPTS = 0

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')

for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

# 실험 결과 리스트 PATH
experiment_path = './report_experiment/'

normal_list = './dataset/week_normal/'
gamble_list = './dataset/week_gamble/'

stock_normal_list = './dataset/total_normal'
stock_gamble_list = './dataset/total_gamble'


normal_file_list = load_dataset_list(normal_list)
gamble_file_list = load_dataset_list(gamble_list)

folder_name = gamble_file_list[0][-10:-4]

normal_total_dataset = load_total_dataframe(normal_file_list)
gamble_total_dataset = load_total_dataframe(gamble_file_list)


while True:

    train, test, total_columns = distribution_dataset(normal_total_dataset, gamble_total_dataset)

    x_train, y_train = train
    x_test, y_test = test

    one_week_model = xgboost.XGBRFClassifier().fit(X=x_train, y=y_train)
    
    y_pred = one_week_model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)

    y_pred = one_week_model.predict(x_test)
    acc_test = accuracy_score(y_pred, y_test)

    if acc_test >= acc_train:
        createFolder(experiment_path + folder_name)
        one_week_model.save_model(experiment_path + folder_name + '/short_model.model')

        # cross_validation
        x_full = np.concatenate((x_train, x_test), axis = 0)
        y_full = np.concatenate((y_train, y_test), axis = 0)
        cross_week_5 = cross_val_score(one_week_model, x_full, y_full, cv=5) # model, train, target, cross validation
        cross_week_10 = cross_val_score(one_week_model, x_full, y_full, cv=10) # model, train, target, cross validation

        break


short_shap_name, short_shap_value = report_shap(x_train, total_columns,  one_week_model)

############ Lite Model 생성

lite_normal_columns = list(set(normal_total_dataset.columns) & set(short_shap_name))
lite_gamble_columns = list(set(gamble_total_dataset.columns) & set(short_shap_name))

while True:

    train, test, total_columns = distribution_dataset(normal_total_dataset[lite_normal_columns], gamble_total_dataset[lite_gamble_columns])

    x_train, y_train = train
    x_test, y_test = test

    one_week_lite_model = xgboost.XGBRFClassifier().fit(X=x_train, y=y_train)


    y_pred = one_week_lite_model.predict(x_train)
    lite_acc_train = accuracy_score(y_pred, y_train)

    y_pred = one_week_lite_model.predict(x_test)
    lite_acc_test = accuracy_score(y_pred, y_test)

    if lite_acc_test >= lite_acc_train :
        createFolder(experiment_path + folder_name)
        one_week_lite_model.save_model(experiment_path + folder_name + '/Lite_short_model.model')

        # cross_validation
        x_full = np.concatenate((x_train, x_test), axis = 0)
        y_full = np.concatenate((y_train, y_test), axis = 0)
        cross_week_lite_5 = cross_val_score(one_week_lite_model, x_full, y_full, cv=5) # model, train, target, cross validation
        cross_week_lite_10 = cross_val_score(one_week_lite_model, x_full, y_full, cv=10) # model, train, target, cross validation

        break

# 파일 이동
get_files = os.listdir(gamble_list)
for g in get_files:
    shutil.move(gamble_list + g, stock_gamble_list)

########### Total Model 생성

normal_file_list = load_dataset_list(stock_normal_list)
gamble_file_list = load_dataset_list(stock_gamble_list)

normal_total_dataset = load_total_dataframe(normal_file_list)
gamble_total_dataset = load_total_dataframe(gamble_file_list)

while True:

    train, test, total_columns = distribution_dataset(normal_total_dataset, gamble_total_dataset)

    x_train, y_train = train
    x_test, y_test = test

    total_model = xgboost.XGBRFClassifier().fit(X=x_train, y=y_train)

    y_pred = total_model.predict(x_train)
    total_acc_train = accuracy_score(y_pred, y_train)

    y_pred = total_model.predict(x_test)
    total_acc_test = accuracy_score(y_pred, y_test)

    if total_acc_test >= total_acc_train:
        createFolder(experiment_path + folder_name)
        total_model.save_model(experiment_path + folder_name + '/total_model.model')

        # cross_validation
        x_full = np.concatenate((x_train, x_test), axis = 0)
        y_full = np.concatenate((y_train, y_test), axis = 0)
        cross_total_5 = cross_val_score(total_model, x_full, y_full, cv=5) # model, train, target, cross validation
        cross_total_10 = cross_val_score(total_model, x_full, y_full, cv=10) # model, train, target, cross validation

        break

total_shap_name, total_shap_value = report_shap(x_train, total_columns,  total_model)

###############################


lite_normal_columns = list(set(normal_total_dataset.columns) & set(total_shap_name))

lite_gamble_columns = list(set(gamble_total_dataset.columns) & set(total_shap_name))

while True:

    train, test, total_columns = distribution_dataset(normal_total_dataset[lite_normal_columns], gamble_total_dataset[lite_gamble_columns])

    x_train, y_train = train
    x_test, y_test = test

    Total_lite_model = xgboost.XGBRFClassifier().fit(X=x_train, y=y_train)


    y_pred = Total_lite_model.predict(x_train)
    total_lite_acc_train = accuracy_score(y_pred, y_train)

    y_pred = Total_lite_model.predict(x_test)
    total_lite_acc_test = accuracy_score(y_pred, y_test)

    if total_lite_acc_test >= total_lite_acc_train :
        createFolder(experiment_path + folder_name)
        Total_lite_model.save_model(experiment_path + folder_name + '/Total_Lite_model.model')

        # cross_validation
        x_full = np.concatenate((x_train, x_test), axis = 0)
        y_full = np.concatenate((y_train, y_test), axis = 0)
        cross_total_lite_5 = cross_val_score(Total_lite_model, x_full, y_full, cv=5) # model, train, target, cross validation
        cross_total_lite_10 = cross_val_score(Total_lite_model, x_full, y_full, cv=10) # model, train, target, cross validation

        break



recent_acc = {'acc_train' : [acc_train],
             'acc_test' : [acc_test] }

short_shap = {'short_shap_name' : short_shap_name,
             'short_shap_value' : short_shap_value }

lite_acc = {'lite_acc_train' : [lite_acc_train],
             'lite_acc_test' : [lite_acc_test] }

total_acc = {'total_acc_train' : [total_acc_train],
             'total_acc_test' : [total_acc_test] }

total_shap = {'total_shap_name' : total_shap_name,
             'total_shap_value' : total_shap_value }

total_lite_acc = {'total_lite_acc_train' : [total_lite_acc_train],
             'total_lite_acc_test' : [total_lite_acc_test] }

cross_acc = {

                'min  cross_week_5' : [np.min(cross_week_5)],
                'max  cross_week_5' : [np.max(cross_week_5)],
                'mean  cross_week_5' : [np.mean(cross_week_5)],

                'min  cross_week_10' : [np.min(cross_week_10)],
                'max  cross_week_10' : [np.max(cross_week_10)],
                'mean  cross_week_10' : [np.mean(cross_week_10)],

                'min  cross_week_lite_5' : [np.min(cross_week_lite_5)],
                'max  cross_week_lite_5' : [np.max(cross_week_lite_5)],
                'mean  cross_week_lite_5' : [np.mean(cross_week_lite_5)],

                'min  cross_week_lite_10' : [np.min(cross_week_lite_10)],
                'max  cross_week_lite_10' : [np.max(cross_week_lite_10)],
                'mean  cross_week_lite_5' : [np.mean(cross_week_lite_10)],

                'min  cross_total_5' : [np.min(cross_total_5)],
                'max  cross_total_5' : [np.max(cross_total_5)],
                'mean  cross_total_5' : [np.mean(cross_total_5)],

                'min  cross_total_10' : [np.min(cross_total_10)],
                'max  cross_total_10' : [np.max(cross_total_10)],
                'mean  cross_total_10' : [np.mean(cross_total_10)],

                'min  cross_total_lite_5' : [np.min(cross_total_lite_5)],
                'max  cross_total_lite_5' : [np.max(cross_total_lite_5)],
                'mean  cross_total_lite_5' : [np.mean(cross_total_lite_5)],

                'min  cross_total_lite_10' : [np.min(cross_total_lite_10)],
                'max  cross_total_lite_10' : [np.max(cross_total_lite_10)],
                'mean  cross_total_lite_10' : [np.mean(cross_total_lite_10)],
            }

recent_acc = pd.DataFrame(recent_acc)

short_shap = pd.DataFrame(short_shap)

lite_acc = pd.DataFrame(lite_acc)

total_acc = pd.DataFrame(total_acc)

total_shap = pd.DataFrame(total_shap)

total_lite_acc = pd.DataFrame(total_lite_acc)

cross_acc = pd.DataFrame(cross_acc)

xlxs_dir = experiment_path + folder_name + '/shap_report.xlsx'

with pd.ExcelWriter(xlxs_dir) as writer:
    
    recent_acc.to_excel(writer, sheet_name = 'recent_acc')

    short_shap.to_excel(writer, sheet_name = 'recent_shap')

    lite_acc.to_excel(writer, sheet_name = 'lite_acc')

    total_acc.to_excel(writer, sheet_name = 'total_acc')

    total_shap.to_excel(writer, sheet_name = 'total_shap')

    total_lite_acc.to_excel(writer, sheet_name = 'total_lite_acc')

    cross_acc.to_excel(writer, sheet_name = 'cross_validation')