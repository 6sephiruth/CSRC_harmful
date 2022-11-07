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

# 220117, 220425, 220502, 220530, 220606, 220613, 220620, 220704, raw_gamble_recent, 221025_labeling
gamble_dataset = pd.read_csv(f"./dataset/week_gamble/221025_labeling.csv")
gamble_dataset = pd.DataFrame(gamble_dataset.drop('Unnamed: 0', axis=1))
# gamble_dataset['label'] = 1

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

total_columns = init_train.columns.drop('label')

x_train = init_train.drop('label', axis=1)
y_train = init_train['label']

x_test = init_test.drop('label', axis=1)
y_test = init_test['label']


##### training #####
try:
    # load model if possible
    model = pickle.load(open('./model/221025.pt','rb'))

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

    pickle.dump(model, open('./model/221025.pt','wb'))




exit()
