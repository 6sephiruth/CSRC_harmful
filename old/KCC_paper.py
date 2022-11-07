from sklearn.metrics import accuracy_score

import tensorflow as tf
import xgboost as xgb
import numpy as np
import pandas as pd

from utils import *

from sklearn.model_selection import cross_val_score

import pickle

seed = 1

# gyu_dates = [220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704]
gyu_dates = [220530, 220606, 220613, 220620, 220704]

for gyu_date in gyu_dates:
    print(gyu_date)

    # 485
    white_dataset = pd.read_csv("./dataset/raw_white.csv")
    white_dataset = pd.DataFrame(white_dataset.drop('Unnamed: 0', axis=1))
    white_dataset['label'] = 0
    # white_dataset['임규민'] = 0


    gamble_dataset = pd.read_csv(f"./dataset/week_gamble/{gyu_date}.csv")
    gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
    gamble_dataset['label'] = 1
    # gamble_dataset['임규민'] = 0

    # white_dataset['임규민'][:int(len(white_dataset)/4)] = 10
    # gamble_dataset['임규민'][:int(len(gamble_dataset)/4*4)] = 10


    # 632
    # ad_dataset = pd.read_csv("./dataset/raw_advertisement.csv")
    # ad_dataset = pd.DataFrame(ad_dataset.drop('Unnamed: 0', axis=1)).iloc[:,:100]
    # ad_dataset['label'] = 2


    white_train = white_dataset.sample(frac=0.8, random_state=seed)
    white_test = white_dataset.drop(white_train.index)

    gamble_train = gamble_dataset.sample(frac=0.8, random_state=seed)
    gamble_test = gamble_dataset.drop(gamble_train.index)

    # ad_train = ad_dataset.sample(frac=0.8, random_state=seed)
    # ad_test = ad_dataset.drop(ad_train.index)

    init_train = pd.concat([white_train, gamble_train])
    init_train.fillna(0, inplace=True)

    init_test = pd.concat([white_test, gamble_test])
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

    x_full = np.concatenate((x_train, x_test), axis = 0)
    y_full = np.concatenate((y_train, y_test), axis = 0)

    cross_week_5 = cross_val_score(model, x_full, y_full, cv=5) # model, train, target, cross validation
    cross_week_10 = cross_val_score(model, x_full, y_full, cv=10) # model, train, target, cross validation

    print('cross')
    print(np.mean(cross_week_5)*100)
    print(np.mean(cross_week_10)*100)

    
    short_shap_name, short_shap_value = report_shap(x_test, total_columns,  model)

    pickle.dump(short_shap_name, open(f'./result/{gyu_date}_name', 'wb'))
    pickle.dump(short_shap_value, open(f'./result/{gyu_date}_value', 'wb'))
