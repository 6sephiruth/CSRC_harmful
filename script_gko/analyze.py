from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import *

from collections import Counter
import operator

import xgboost as xgb
import pandas as pd
import numpy as np

import time
import shap

def get_invar(df, drop_field=[]):
    df_new = df[df.columns.drop(drop_field)]

    return list(df_new.columns[df_new.max(axis=0) == df_new.min(axis=0)])

def identity(df, a, b):
    return df

def minmax(df, mi, mx):
    ndf = df.copy()
    for c in df.columns:
        if mi[c] == mx[c]:
            ndf[c] = df[c] - mi[c]
        else:
            ndf[c] = (df[c] - mi[c]) / (mx[c] - mi[c])
    return ndf

def zscore(df, mean, std):
    ndf = df.copy()
    for c in df.columns:
        if std[c] == 0:
            ndf[c] = df[c] - mean[c]
        else:
            ndf[c] = (df[c] - mean[c]) / std[c]
    return ndf

def aug_norm(x, sigma=0.03):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def aug_pois(x, lam):
    return x + np.random.poisson(lam=lam, size=x.shape)

def aug_mixup(x, y, lamb):
    zeros = y == 0
    ones = y == 1
    cols = x.columns

    x_aug = np.zeros_like(x)
    x_aug[zeros] = mixup(x[zeros], lamb)
    x_aug[ones] = mixup(x[ones], lamb)

    return pd.DataFrame(x_aug, columns=cols)

def mixup(x, lamb=0.5):
    x_rand = x.sample(frac=1)
    return lamb*x + (1-lamb)*x_rand

def get_stats(y_t, y_p, weight=None):
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, sample_weight=weight).ravel()

    acc = (tp+tn)/(tn+fp+fn+tp)
    tpr = tp/(fp+tp)
    tnr = tn/(fn+tn)

    return np.array([acc,tpr,tnr])

def load_df(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df.drop('Unnamed: 0', axis=1))

    # remove all zeros
    s = np.sum(df, axis=1)

    return df[s > 0]

def train_test_split(df, frac=0.8, seed=0):
    train = df.sample(frac=frac, random_state=seed)
    test = df.drop(train.index)

    return train, test

def get_keywords(xs, ys, explainer, n_key=5, w_key=3):
    shap_val = explainer.shap_values(xs, check_additivity=False)
    best_idx = np.argsort(shap_val)[...,-int(w_key*n_key):]

    key_all = np.array(xs.columns)[best_idx]
    key_label = [key_all[y,i] for i,y in enumerate(ys)]

    return np.array(key_label)

def make_pattern(xs, ys, explainer, n_key=5):
    keywords = get_keywords(xs, ys, explainer, n_key, w_key=2)

    pattern = []
    for l in set(ys):
        idx_l = ys == l
        key_l = keywords[idx_l]   # extract keywords

        # extract most frequent keywords
        cnt_l = Counter(list(key_l.flatten()))
        freq_l = sorted(cnt_l.items(), key=operator.itemgetter(1), reverse=True)[:n_key]

        ptn_l = []
        for k,_ in freq_l:
            avg_all = np.mean(xs[k])
            avg_l = np.mean(xs[idx_l][k])
            sgn_k = avg_l >= avg_all

            ptn_l.append((k,sgn_k))

        pattern.append(ptn_l)

    return pattern

def get_pattern(xs, ys, xs_mean, explainer, n_key=5):
    keywords = get_keywords(xs, ys, explainer, n_key, w_key=1)

    pattern = []
    for (_,x),ks in zip(xs.iterrows(),keywords):
        ptn = [(k,x[k]>=xs_mean[k]) for k in ks]
        pattern.append(ptn)

    return pattern

def pattern_score(ps, ys, ref_p):
    assert len(ps) == len(ys)
    assert len(ref_p) == len(set(ys))

    return [match_pattern(p,ref_p[y]) for p,y in zip(ps,ys)]

def match_pattern(p1, p2):
    # TODO: weighted scoring? according to ordering
    assert len(p1) == len(p2)

    score = 0
    for k,s in p1:
        if (k,s) in p2:
            score += 1
        elif (k,not s) in p2:
            score += 0.5

    return score


DATA_DIR = '../dataset/'

### main
def main():
    # set exp arguments
    bench_name = 'result.tsv'
    model_name = 'RandomForestClassifier'
    cls_field = 'label'

    # setup initial dataset
    df_gam = load_df(f'{DATA_DIR}/week_gamble/raw_gamble_recent.csv')
    df_adv = load_df(f'{DATA_DIR}/raw_advertisement.csv')
    df_etc = load_df(f'{DATA_DIR}/raw_white.csv')

    df_gam[cls_field] = 0         # 도박
    df_adv[cls_field] = 1         # 광고
    df_etc[cls_field] = 2         # 기타

    # all data
    data_raw = pd.concat([df_gam, df_adv, df_etc])
    data_raw.fillna(0, inplace=True)

    weight = data_raw.value_counts(normalize=True, sort=False)

    # remove duplicates prior to split
    data_raw.drop_duplicates(inplace=True)
    data_raw['weight'] = weight.values

    # separate labels
    gam_raw = data_raw[data_raw[cls_field] == 0]
    adv_raw = data_raw[data_raw[cls_field] == 1]
    etc_raw = data_raw[data_raw[cls_field] == 2]

    seed = 8
    frac = 0.5

    # randomly split between labels
    gam_tr, gam_te = train_test_split(gam_raw, frac=frac, seed=seed)
    adv_tr, adv_te = train_test_split(adv_raw, frac=frac, seed=seed)
    etc_tr, etc_te = train_test_split(etc_raw, frac=frac, seed=seed)

    train_raw = pd.concat([gam_tr, adv_tr, etc_tr])
    test_raw = pd.concat([gam_te, adv_te, etc_te])

    # preprocessing
    invar_fields = get_invar(train_raw, drop_field=[cls_field])
    drop = invar_fields + ['weight', cls_field]
    valid_cols = train_raw.columns.drop(drop)

    # configure dataset and weights
    x_train = train_raw[valid_cols]
    y_train = train_raw[cls_field]
    w_train = train_raw['weight']
    x_test = test_raw[valid_cols]
    y_test = test_raw[cls_field]
    w_test = test_raw['weight']

    # training
    print('[*] training ...')
    model = eval(model_name)(random_state=seed, n_jobs=-1)

    s = time.time()
    model.fit(x_train, y_train)
    e = time.time()

    print('[*]', e-s, 'seconds taken')

    # evaluation
    y_pred_tr = model.predict(x_train)
    acc_tr = accuracy_score(y_train, y_pred_tr, sample_weight=w_train)

    y_pred_te = model.predict(x_test)
    acc_te = accuracy_score(y_test, y_pred_te, sample_weight=w_test)

    print(acc_tr, acc_te)

    # number of keywords to extract
    N_KEY = 15

    # explainer
    explainer = shap.TreeExplainer(model, seed=seed)

    ref_pattern = make_pattern(xs=x_train,
                               ys=y_train,
                               explainer=explainer,
                               n_key=N_KEY)

    train_pattern = get_pattern(xs=x_train,
                                ys=y_train,
                                xs_mean=x_train.mean(),
                                explainer=explainer,
                                n_key=N_KEY)

    print(ref_pattern)
    exit()

    s_train = pattern_score(train_pattern, y_train, ref_pattern)
    print(np.average(s_train, weights=w_train))
    print(np.max(s_train))
    print(np.min(s_train))

    wrong_idx = y_test != y_pred_te
    wrong_pattern = get_pattern(xs=x_test[wrong_idx],
                                ys=y_pred_te[wrong_idx],
                                xs_mean=x_train.mean(),
                                explainer=explainer,
                                n_key=N_KEY)
    s_wrong = pattern_score(wrong_pattern, y_pred_te[wrong_idx], ref_pattern)
    print(np.average(s_wrong, weights=w_test[wrong_idx]))
    print(np.max(s_wrong))
    print(np.min(s_wrong))

    corr_idx = y_test == y_pred_te
    corr_pattern = get_pattern(xs=x_test[corr_idx],
                               ys=y_pred_te[corr_idx],
                               xs_mean=x_train.mean(),
                               explainer=explainer,
                               n_key=N_KEY)
    s_corr = pattern_score(corr_pattern, y_pred_te[corr_idx], ref_pattern)
    print(np.average(s_corr, weights=w_test[corr_idx]))
    print(np.max(s_corr))
    print(np.min(s_corr))

    exit()

    std = x_train.std()

    # set random seed
    for n_aug in range(1,20):

        x_a, y_a, w_a = [], [], []
        for _ in range(n_aug):
            # augmentation
            #x_aug = aug_mixup(x_train, y_train, lamb=0.9)
            x_aug = x_train           # do not apply mixup
            x_aug = aug_norm(x_aug, sigma=std/50)
            y_aug = y_train.copy()
            w_aug = w_train.copy()

            x_a.append(x_aug)
            y_a.append(y_aug)
            w_a.append(w_aug)

        # concatenation
        x_train_ = pd.concat([x_train] + x_a)
        y_train_ = pd.concat([y_train] + y_a)
        w_train_ = pd.concat([w_train] + w_a)

        # training
        model_ = eval(model_name)(random_state=seed)
        model_.fit(x_train_, y_train_)

        # evaluation
        y_pred_tr = model_.predict(x_train_)
        a_tr = get_stats(y_train_, y_pred_tr, weight=w_train_)[0]

        y_pred_te = model_.predict(x_test)
        a_te = get_stats(y_test, y_pred_te, weight=w_test)[0]

        y_pred_p = model_.predict_proba(x_test)
        a_loss = log_loss(y_test, y_pred_p, sample_weight=w_test)

        # benchmark
        info = [model_name, str(frac), str(n_aug)]

        with open(bench_name,'a') as f:
            s = '\t'.join([model_name, str(n_aug), normalize]) + '\t'
            #s += '\t'.join(list(map(lambda t: f"{t:.4f}", b_te))) + '\t'
            #s += '\t'.join(list(map(lambda t: f"{t:.4f}", a_te))) + '\n'
            s += '\t'.join(list(map(lambda t: f"{t:.4f}",
                            [b_te, a_te, b_loss, a_loss]))) + '\n'
            f.write(s)


# run main
if __name__ == "__main__":
    main()